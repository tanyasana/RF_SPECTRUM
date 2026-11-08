"""
=============================================================================
Module: test_dashboard_full_execution.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 7 - Headless Simulation of Complete Dashboard User Interactions
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from python.fft_analysis import compute_fft, compute_spectrum_occupancy, load_iq_from_csv
from python.spectrogram import compute_spectrogram
from python.anomaly_detection import compute_statistical_rf_metrics
from python.signal_generator import (
    generate_bpsk, generate_qpsk, generate_16qam,
    add_awgn_noise, add_power_burst, add_frequency_offset
)
from scripts.run_spectrum_top_simulation import SpectrumTopSimulator
from ai_model.classifier import RFSignalClassifier
from ai_model.dataset_builder import compute_differential_phase_features, generate_symbols_for_class


def test_scenario(name, i_samples, q_samples, hw_thresh_float=2.5, hw_limit=5):
    print(f"\n--- Testing Scenario: {name} ({len(i_samples)} samples) ---")
    complex_sig = i_samples + 1j * q_samples
    num_samples = len(complex_sig)

    # 1. FFT
    freqs, psd_db, _ = compute_fft(complex_sig, sample_rate=1e6, window="hann")
    occ, occ_bins, total_bins = compute_spectrum_occupancy(psd_db, threshold_db=-20.0)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs/1e3, y=psd_db, mode='lines'))
    _ = fig_fft.to_json()
    print("  [OK] FFT plot generated successfully.")

    # 2. Spectrogram (Transposed)
    s_freqs, s_times, s_psd = compute_spectrogram(complex_sig, sample_rate=1e6, nperseg=128, noverlap=64)
    fig_spec = go.Figure(data=go.Heatmap(z=s_psd.T, x=s_freqs/1e3, y=s_times*1e3))
    _ = fig_spec.to_json()
    print(f"  [OK] Spectrogram plot generated successfully (shape: {s_psd.T.shape}).")

    # 3. Constellation & Diff Phase
    d_i, d_q = compute_differential_phase_features(i_samples, q_samples)
    fig_iq = go.Figure(go.Scatter(x=i_samples, y=q_samples, mode='markers'))
    fig_diff = go.Figure(go.Scatter(x=d_i, y=d_q, mode='markers'))
    _ = fig_iq.to_json()
    _ = fig_diff.to_json()
    print("  [OK] Constellation & Diff phase plots generated successfully.")

    # 4. AI Classifier
    classifier = RFSignalClassifier()
    ai_res = classifier.predict_signal(i_samples, q_samples, seq_len=128, power_threshold=hw_thresh_float)
    fig_probs = go.Figure(go.Bar(x=list(ai_res['probabilities'].keys()), y=list(ai_res['probabilities'].values())))
    _ = fig_probs.to_json()
    print(f"  [OK] AI Inference: {ai_res['predicted_class']} ({ai_res['confidence_pct']}%) | Threat: {ai_res['is_anomaly']}")

    # 5. Verilog Emulation (Cycle by Cycle)
    scale = 16384.0
    hw_fixed_thresh = int(hw_thresh_float * (scale ** 2))
    sim = SpectrumTopSimulator(threshold=hw_fixed_thresh, anomaly_limit=hw_limit)
    sim.reset()

    hw_cycles = []
    hw_powers = []
    hw_hits = []
    hw_counts = []
    hw_states = []

    emulation_len = min(num_samples, 1024)
    for idx in range(emulation_len):
        in_i = int(np.clip(i_samples[idx] * scale, -32768, 32767))
        in_q = int(np.clip(q_samples[idx] * scale, -32768, 32767))
        step = sim.clock_cycle(in_i, in_q, sample_valid_in=True)

        hw_cycles.append(idx + 1)
        hw_powers.append(step["power"])
        hw_hits.append(1 if step["threshold_hit"] else 0)
        hw_counts.append(step["abnormal_count"])
        hw_states.append(step["fsm_state"])

    fig_hw = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig_hw.add_trace(go.Scatter(x=hw_cycles, y=hw_powers), row=1, col=1)
    fig_hw.add_trace(go.Scatter(x=hw_cycles, y=hw_hits), row=2, col=1)
    fig_hw.add_trace(go.Scatter(x=hw_cycles, y=hw_counts), row=3, col=1)
    _ = fig_hw.to_json()

    df_trace = pd.DataFrame({
        "Cycle": hw_cycles,
        "Power (I2+Q2)": hw_powers,
        "Threshold Hit": hw_hits,
        "Counter Value": hw_counts,
        "FSM State": hw_states
    })
    _ = df_trace.to_csv(index=False)
    print(f"  [OK] Verilog hardware subplots and DataFrame trace generated ({len(df_trace)} rows, violations={sum(hw_hits)}).")


def run_all_checks():
    print("=" * 80)
    print("STARTING COMPREHENSIVE DASHBOARD TEST EXECUTION")
    print("=" * 80)

    # Pre-generated datasets
    ds_list = [
        "clean_qpsk.csv",
        "noisy_bpsk_snr6db.csv",
        "clean_16qam.csv",
        "burst_interference.csv",
        "swept_chirp_jammed.csv"
    ]
    for ds in ds_list:
        p = os.path.join(base_dir, "datasets", "simulated", ds)
        df = pd.read_csv(p)
        test_scenario(ds, df["I"].to_numpy().astype(np.float32), df["Q"].to_numpy().astype(np.float32))

    # Synthesized signals
    for mod in ["QPSK", "BPSK", "8PSK", "16-QAM"]:
        if mod == "BPSK":
            sig = generate_bpsk(1024, 8, 15.0)
        elif mod == "QPSK":
            sig = generate_qpsk(1024, 8, 15.0)
        elif mod == "8PSK":
            syms = generate_symbols_for_class(2, 132)
            sig = add_awgn_noise(np.repeat(syms, 8)[:1024], 15.0)
        else:
            sig = generate_16qam(1024, 8, 15.0)

        # with burst
        sig_burst = add_power_burst(sig, start_idx=400, duration=100, amplitude_multiplier=5.0)
        test_scenario(f"Synthesized {mod} + Burst", np.real(sig_burst).astype(np.float32), np.imag(sig_burst).astype(np.float32))

    print("\n" + "=" * 80)
    print("ALL DASHBOARD SCENARIOS & PLOTS EXECUTED WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_checks()
