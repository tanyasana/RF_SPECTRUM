"""
=============================================================================
Application: app.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 7 - Interactive Web Dashboard & Real-Time Operator Console (Audited)
=============================================================================

This Streamlit application serves as the central Human-Machine Interface (HMI)
for our Hardware-Software Co-Design system, unifying:
  1. Python Digital Signal Processing (FFT, Welch PSD, Waterfall Spectrogram).
  2. Synthesizable Verilog Digital Circuitry (Cycle-by-cycle telemetry & FSM).
  3. PyTorch Deep Learning 1D-CNN (Differential Phase AMC & Jamming Gate).
  4. Statistical RF Metric Engine (PAPR, Kurtosis, Skewness, Shannon Entropy).
"""

import os
import sys
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Dynamic path resolution to root project directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from python.fft_analysis import compute_fft, compute_spectrum_occupancy, load_iq_from_csv
from python.spectrogram import compute_spectrogram
from python.anomaly_detection import compute_statistical_rf_metrics
from python.signal_generator import (
    generate_bpsk, generate_qpsk, generate_16qam,
    add_awgn_noise, add_power_burst, add_frequency_offset
)
from scripts.run_spectrum_top_simulation import SpectrumTopSimulator
from ai_model.classifier import RFSignalClassifier
from ai_model.dataset_builder import compute_differential_phase_features, generate_symbols_for_class, LABEL_MAP


# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & DARK CYBER THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Driven RF Spectrum Monitoring System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Top Header Card */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .main-header h1 {
        color: #38BDF8;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 14px;
        margin: 0;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .kpi-title {
        color: #9CA3AF;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Status Badges */
    .badge-normal {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #059669;
        border-radius: 6px;
        padding: 4px 10px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-threat {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid #DC2626;
        border-radius: 6px;
        padding: 4px 10px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-alert {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid #D97706;
        border-radius: 6px;
        padding: 4px 10px;
        font-weight: 700;
        display: inline-block;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0F172A;
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94A3B8;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED MODEL & DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_ai_classifier():
    """Loads and caches the 1D-CNN RF Classifier model checkpoint."""
    weights_path = os.path.join(ROOT_DIR, "ai_model", "weights", "best_rf_classifier.pt")
    if not os.path.exists(weights_path):
        return None
    classifier = RFSignalClassifier(weights_path=weights_path, device="cpu")
    return classifier


def generate_8psk_signal(num_samples: int = 1024, samples_per_symbol: int = 8, snr_db: float = None) -> np.ndarray:
    """Helper to synthesize 8PSK for interactive mode."""
    num_symbols = int(np.ceil(num_samples / samples_per_symbol)) + 4
    symbols = generate_symbols_for_class(class_id=2, num_symbols=num_symbols)
    raw_sig = np.repeat(symbols, samples_per_symbol)[:num_samples]
    if snr_db is not None:
        raw_sig = add_awgn_noise(raw_sig, snr_db)
    return raw_sig


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & INPUT CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
    st.markdown("### 🎛️ Operator Controls")
    
    data_source = st.radio(
        "Signal Input Mode:",
        ["Pre-Generated Datasets", "Upload Custom CSV", "Live Interactive Synthesis"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ Hardware Parameters")
    hw_threshold_float = st.slider(
        "Power Detection Threshold (x Normalized RMS)",
        min_value=1.2, max_value=10.0, value=2.5, step=0.1,
        help="Signals whose instantaneous power exceeds this factor relative to normalized RMS trip the Verilog comparator."
    )
    hw_anomaly_limit = st.slider(
        "Consecutive Violation Limit (Cycles)",
        min_value=1, max_value=20, value=5, step=1,
        help="Number of consecutive clock cycles exceeding threshold before the hardware counter fires an anomaly limit alert."
    )
    
    st.markdown("---")
    st.markdown("#### 🔬 DSP Parameters")
    fft_window = st.selectbox("FFT Window Function:", ["hann", "hamming", "blackman", "rect"], index=0)
    stft_nperseg = st.selectbox("STFT Resolution (Samples):", [64, 128, 256], index=1)
    
    st.markdown("---")
    st.caption("AI-Driven RF Spectrum Monitoring SoC | Capstone Architecture")


# -----------------------------------------------------------------------------
# 4. SIGNAL INGESTION LOGIC
# -----------------------------------------------------------------------------
signal_name = "Custom Signal"
i_samples = np.array([], dtype=np.float32)
q_samples = np.array([], dtype=np.float32)

if data_source == "Pre-Generated Datasets":
    dataset_options = {
        "Clean QPSK (Normal Transmission)": "clean_qpsk.csv",
        "Noisy BPSK (SNR = 6 dB)": "noisy_bpsk_snr6db.csv",
        "Clean 16-QAM (Multi-Tier Amplitude)": "clean_16qam.csv",
        "Pulsed Burst Interference (Energy Surge)": "burst_interference.csv",
        "Swept Chirp Jammer (Broadband Threat)": "swept_chirp_jammed.csv"
    }
    selected_name = st.sidebar.selectbox("Select Signal Scenario:", list(dataset_options.keys()))
    signal_name = selected_name
    csv_path = os.path.join(ROOT_DIR, "datasets", "simulated", dataset_options[selected_name])
    
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        i_samples = df_csv["I"].to_numpy().astype(np.float32)
        q_samples = df_csv["Q"].to_numpy().astype(np.float32)
    else:
        st.error(f"Dataset file not found at: {csv_path}")

elif data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload IQ CSV File (Columns: I, Q)", type=["csv"])
    if uploaded_file is not None:
        df_up = pd.read_csv(uploaded_file)
        if "I" in df_up.columns and "Q" in df_up.columns:
            i_samples = df_up["I"].to_numpy().astype(np.float32)
            q_samples = df_up["Q"].to_numpy().astype(np.float32)
            signal_name = uploaded_file.name
        else:
            st.error("Uploaded CSV must contain 'I' and 'Q' header columns!")
    else:
        st.info("Please upload an IQ CSV file with columns 'I' and 'Q'.")

elif data_source == "Live Interactive Synthesis":
    synth_mod = st.sidebar.selectbox("Modulation Format:", ["QPSK", "BPSK", "8PSK", "16-QAM"])
    synth_snr = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB):", min_value=-6.0, max_value=30.0, value=15.0, step=1.0)
    synth_len = st.sidebar.slider("Sample Count:", min_value=256, max_value=2048, value=1024, step=128)
    inject_burst = st.sidebar.checkbox("Inject Rogue Jammer Burst", value=False)
    
    signal_name = f"Synthesized {synth_mod} (SNR={synth_snr}dB)"
    if synth_mod == "BPSK":
        sig = generate_bpsk(num_samples=synth_len, samples_per_symbol=8, snr_db=synth_snr)
    elif synth_mod == "QPSK":
        sig = generate_qpsk(num_samples=synth_len, samples_per_symbol=8, snr_db=synth_snr)
    elif synth_mod == "8PSK":
        sig = generate_8psk_signal(num_samples=synth_len, samples_per_symbol=8, snr_db=synth_snr)
    else:
        sig = generate_16qam(num_samples=synth_len, samples_per_symbol=8, snr_db=synth_snr)
        
    if inject_burst:
        sig = add_power_burst(sig, start_idx=int(synth_len*0.4), duration=100, amplitude_multiplier=5.0)
        signal_name += " + Jamming Burst"
        
    i_samples = np.real(sig).astype(np.float32)
    q_samples = np.imag(sig).astype(np.float32)


# -----------------------------------------------------------------------------
# 5. CORE COMPUTATIONS (DSP, AI INFERENCE & HARDWARE EMULATION)
# -----------------------------------------------------------------------------
if len(i_samples) > 0 and len(q_samples) > 0:
    complex_signal = i_samples + 1j * q_samples
    num_samples = len(complex_signal)
    sample_rate = 1e6  # 1.0 MHz sampling rate
    
    # 5.1 FFT & Spectrum Occupancy
    freqs, psd_db, _ = compute_fft(complex_signal, sample_rate=sample_rate, window=fft_window)
    occupancy_pct, occupied_mask, eff_thresh = compute_spectrum_occupancy(psd_db, threshold_db=-20.0)
    occupied_bins = int(np.count_nonzero(occupied_mask))
    total_bins = len(psd_db)
    occupied_bw_khz = float((occupancy_pct / 100.0) * (sample_rate / 1e3))
    
    # 5.2 Statistical Metrics
    stat_metrics = compute_statistical_rf_metrics(i_samples, q_samples)
    
    # 5.3 AI Model Inference
    classifier = get_ai_classifier()
    if classifier is not None:
        ai_res = classifier.predict_signal(i_samples, q_samples, seq_len=128, power_threshold=hw_threshold_float)
    else:
        ai_res = {
            "predicted_class": "1D-CNN Checkpoint Loading...",
            "confidence_pct": 0.0,
            "probabilities": {"BPSK": 25.0, "QPSK": 25.0, "8PSK": 25.0, "16-QAM": 25.0},
            "shannon_entropy": 2.0,
            "is_anomaly": False,
            "anomaly_reason": "Weights not loaded"
        }
        
    # 5.4 Verilog Hardware Emulation with Corrected Q1.15 Fixed-Point Power Scaling
    # In Q1.15 fixed-point, 1.0 = 16384 (or 32767).
    # Power = I^2 + Q^2. For float unit power P=1.0, fixed power = (16384)^2 = 268,435,456.
    # Therefore, hardware threshold in fixed units = hw_threshold_float * (16384^2).
    scale = 16384.0
    scale_sq = scale ** 2
    hw_fixed_thresh = int(hw_threshold_float * scale_sq)
    
    sim = SpectrumTopSimulator(threshold=hw_fixed_thresh, anomaly_limit=hw_anomaly_limit)
    sim.reset()
    
    # Run cycle-by-cycle Verilog emulation
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
        
    final_fsm_state = hw_states[-1] if hw_states else "IDLE"
    total_violations = sum(hw_hits)

    # -------------------------------------------------------------------------
    # 6. HEADER & LIVE KPI DASHBOARD CARDS
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div class="main-header">
        <h1>📡 AI-Driven RF Spectrum Monitoring & Interference Detection System</h1>
        <p>Active Ingestion: <b>{signal_name}</b> | Dual Domain Hardware-Software Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Frame Size</div>
            <div class="kpi-value" style="color: #38BDF8;">{num_samples} <span style="font-size: 13px; color: #64748B;">IQ</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        conf_color = "#34D399" if ai_res['confidence_pct'] >= 90.0 else "#FBBF24"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">AI Classification</div>
            <div class="kpi-value" style="color: {conf_color};">{ai_res['predicted_class']} <span style="font-size: 12px; color: #94A3B8;">({ai_res['confidence_pct']:.1f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        if ai_res['is_anomaly'] or final_fsm_state in ["2'b10 (ALERT)", "2'b11 (REPORT)"]:
            badge_html = '<span class="badge-threat">THREAT / JAMMER</span>'
        else:
            badge_html = '<span class="badge-normal">NORMAL</span>'
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Threat Status</div>
            <div style="margin-top: 4px;">{badge_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        state_color = "#F87171" if "ALERT" in final_fsm_state or "REPORT" in final_fsm_state else "#38BDF8"
        clean_state = final_fsm_state.split("(")[-1].replace(")", "")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Verilog FSM State</div>
            <div class="kpi-value" style="color: {state_color};">{clean_state}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Spectrum Occupancy</div>
            <div class="kpi-value" style="color: #A855F7;">{occupancy_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. TABBED VISUALIZATION CONSOLE
    # -------------------------------------------------------------------------
    tab_fft, tab_waterfall, tab_const, tab_ai, tab_verilog = st.tabs([
        "📊 Spectral Analysis (FFT)",
        "🌊 2D Waterfall Spectrogram",
        "🎯 IQ & Phase Space",
        "🧠 AI Modulation Classifier",
        "⚡ Verilog Hardware Telemetry"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: SPECTRAL ANALYSIS (FFT & PSD)
    # -------------------------------------------------------------------------
    with tab_fft:
        st.subheader("1024-Point Fast Fourier Transform (FFT) Power Spectrum")
        
        freq_khz = freqs / 1e3
        fig_fft = go.Figure()
        
        # Power Spectrum Curve
        fig_fft.add_trace(go.Scatter(
            x=freq_khz, y=psd_db,
            mode='lines',
            name='PSD (dBFS)',
            line=dict(color='#0284C7', width=1.8),
            fill='tozeroy',
            fillcolor='rgba(2, 132, 199, 0.12)'
        ))
        
        # Energy Threshold Line (-20 dBFS)
        fig_fft.add_hline(
            y=-20.0,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="Occupancy Cutoff Threshold (-20 dBFS)",
            annotation_position="top right"
        )
        
        fig_fft.update_layout(
            paper_bgcolor='#0F172A',
            plot_bgcolor='#111827',
            font=dict(color='#E2E8F0'),
            xaxis=dict(title='Frequency [kHz]', gridcolor='#1E293B', zerolinecolor='#334155'),
            yaxis=dict(title='Power Spectral Density [dBFS]', gridcolor='#1E293B', range=[-80, 5]),
            margin=dict(l=40, r=40, t=30, b=40),
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig_fft, use_container_width=True)
        
        # Spectral Info Columns
        sc1, sc2, sc3, sc4 = st.columns(4)
        peak_idx = int(np.argmax(psd_db))
        peak_freq = freq_khz[peak_idx]
        sc1.metric("Peak Frequency", f"{peak_freq:.1f} kHz")
        sc2.metric("Occupied Bandwidth", f"{occupied_bw_khz:.1f} kHz")
        sc3.metric("Noise Floor Est.", f"{np.percentile(psd_db, 10):.1f} dBFS")
        sc4.metric("Occupancy Ratio", f"{occupancy_pct:.1f}%")

    # -------------------------------------------------------------------------
    # TAB 2: 2D WATERFALL SPECTROGRAM (STFT - ORIENTATION CORRECTED)
    # -------------------------------------------------------------------------
    with tab_waterfall:
        st.subheader("Time-Frequency Waterfall Spectrogram (Short-Time Fourier Transform)")
        
        s_freqs, s_times, s_psd = compute_spectrogram(complex_signal, sample_rate=sample_rate, nperseg=stft_nperseg, noverlap=stft_nperseg//2)
        s_freq_khz = s_freqs / 1e3
        s_time_ms = s_times * 1e3
        
        # Correctly transpose s_psd so rows map to time (y) and columns map to frequency (x)
        fig_spec = go.Figure(data=go.Heatmap(
            z=s_psd.T,
            x=s_freq_khz,
            y=s_time_ms,
            colorscale='Viridis',
            colorbar=dict(title='dBFS')
        ))
        
        fig_spec.update_layout(
            paper_bgcolor='#0F172A',
            plot_bgcolor='#111827',
            font=dict(color='#E2E8F0'),
            xaxis=dict(title='Frequency [kHz]', gridcolor='#1E293B'),
            yaxis=dict(title='Time [ms]', gridcolor='#1E293B'),
            margin=dict(l=40, r=40, t=30, b=40),
            height=450
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        st.caption("ℹ️ Swept chirps appear as diagonal trajectories; pulsed interference manifests as horizontal energy flashes.")

    # -------------------------------------------------------------------------
    # TAB 3: IQ CONSTELLATION & DIFFERENTIAL PHASE SPACE
    # -------------------------------------------------------------------------
    with tab_const:
        st.subheader("IQ Constellation & Differential Phase Orthogonality")
        
        d_i, d_q = compute_differential_phase_features(i_samples, q_samples)
        
        c_col1, c_col2 = st.columns(2)
        
        with c_col1:
            st.markdown("##### Standard IQ Constellation (Q vs. I)")
            fig_iq = go.Figure()
            fig_iq.add_trace(go.Scatter(
                x=i_samples, y=q_samples,
                mode='markers',
                marker=dict(size=5, color='#38BDF8', opacity=0.7),
                name='Raw IQ'
            ))
            fig_iq.update_layout(
                paper_bgcolor='#0F172A',
                plot_bgcolor='#111827',
                font=dict(color='#E2E8F0'),
                xaxis=dict(title='In-Phase (I)', gridcolor='#1E293B', range=[-3.5, 3.5]),
                yaxis=dict(title='Quadrature (Q)', gridcolor='#1E293B', range=[-3.5, 3.5], scaleanchor="x", scaleratio=1),
                margin=dict(l=30, r=30, t=20, b=30),
                height=380
            )
            st.plotly_chart(fig_iq, use_container_width=True)
            
        with c_col2:
            st.markdown("##### Differential Phase Space (ΔQ vs. ΔI)")
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(
                x=d_i, y=d_q,
                mode='markers',
                marker=dict(size=5, color='#34D399', opacity=0.7),
                name='Differential'
            ))
            fig_diff.update_layout(
                paper_bgcolor='#0F172A',
                plot_bgcolor='#111827',
                font=dict(color='#E2E8F0'),
                xaxis=dict(title='Differential I (cos Δφ)', gridcolor='#1E293B', range=[-1.5, 1.5]),
                yaxis=dict(title='Differential Q (sin Δφ)', gridcolor='#1E293B', range=[-1.5, 1.5], scaleanchor="x", scaleratio=1),
                margin=dict(l=30, r=30, t=20, b=30),
                height=380
            )
            st.plotly_chart(fig_diff, use_container_width=True)
            
        st.info("💡 **Phase Invariance Proof:** Notice how the Differential Phase Space organizes symbol jumps into orthogonal coordinates `{0, ±1}`, completely canceling static carrier rotation angles (θ₀).")

    # -------------------------------------------------------------------------
    # TAB 4: AI CLASSIFICATION & THREAT DIAGNOSTICS
    # -------------------------------------------------------------------------
    with tab_ai:
        st.subheader("Deep Learning 1D-CNN Automatic Modulation Classification")
        
        ai_col1, ai_col2 = st.columns([3, 2])
        
        with ai_col1:
            st.markdown("##### Multi-Class Softmax Probability Distribution")
            classes = list(ai_res['probabilities'].keys())
            probs = list(ai_res['probabilities'].values())
            
            bar_colors = ['#10B981' if c == ai_res['predicted_class'] else '#334155' for c in classes]
            
            fig_probs = go.Figure(go.Bar(
                x=classes, y=probs,
                marker_color=bar_colors,
                text=[f"{p:.1f}%" for p in probs],
                textposition='auto'
            ))
            fig_probs.update_layout(
                paper_bgcolor='#0F172A',
                plot_bgcolor='#111827',
                font=dict(color='#E2E8F0'),
                yaxis=dict(title='Confidence [%]', range=[0, 105], gridcolor='#1E293B'),
                margin=dict(l=30, r=30, t=20, b=30),
                height=320
            )
            st.plotly_chart(fig_probs, use_container_width=True)
            
        with ai_col2:
            st.markdown("##### Threat Diagnostic Engine")
            st.metric("Shannon Prediction Entropy (H)", f"{ai_res['shannon_entropy']:.3f} bits")
            st.metric("Peak-to-Average Power Ratio (PAPR)", f"{stat_metrics['papr_db']:.2f} dB")
            st.metric("Spectral Kurtosis", f"{stat_metrics['kurtosis']:.2f}")
            
            if ai_res['is_anomaly']:
                st.error(f"🚨 **Threat Alert:** {ai_res['anomaly_reason']}")
            else:
                st.success("✅ **Transmission Clear:** Normal signal within validated statistical bounds.")

    # -------------------------------------------------------------------------
    # TAB 5: VERILOG HARDWARE TELEMETRY & SOC EMULATION (CORRECTED SCALING)
    # -------------------------------------------------------------------------
    with tab_verilog:
        st.subheader("Synthesizable Verilog Digital Circuitry Telemetry (Cycle-by-Cycle)")
        st.caption("Verilog Architecture: `iq_register` ➡️ `magnitude_calculator` ➡️ `threshold_detector` ➡️ `abnormal_counter` ➡️ `detection_fsm`")
        
        # Hardware Telemetry Subplots
        fig_hw = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[
                f"Pipelined Power Calculator (P = I² + Q²) [Thresh = {hw_fixed_thresh:,}]",
                "Threshold Detector Hit Train (Boolean Flag)",
                f"Abnormal Event Counter Accumulator [Limit = {hw_anomaly_limit}]"
            ]
        )
        
        # Trace 1: Power vs Threshold
        fig_hw.add_trace(
            go.Scatter(x=hw_cycles, y=hw_powers, name="Power (I²+Q²)", line=dict(color='#0284C7', width=1.5)),
            row=1, col=1
        )
        fig_hw.add_hline(y=hw_fixed_thresh, line_dash="dash", line_color="#EF4444", row=1, col=1)
        
        # Trace 2: Hits
        fig_hw.add_trace(
            go.Scatter(x=hw_cycles, y=hw_hits, name="Threshold Hit", line=dict(color='#F59E0B', width=1.2)),
            row=2, col=1
        )
        
        # Trace 3: Counter
        fig_hw.add_trace(
            go.Scatter(x=hw_cycles, y=hw_counts, name="Violation Count", line=dict(color='#EC4899', width=1.8)),
            row=3, col=1
        )
        fig_hw.add_hline(y=hw_anomaly_limit, line_dash="dash", line_color="#DC2626", row=3, col=1)
        
        fig_hw.update_layout(
            paper_bgcolor='#0F172A',
            plot_bgcolor='#111827',
            font=dict(color='#E2E8F0'),
            height=550,
            margin=dict(l=40, r=40, t=40, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_hw, use_container_width=True)
        
        # Hardware Summary Metrics
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Emulated Cycles", len(hw_cycles))
        h2.metric("Threshold Violations", total_violations)
        h3.metric("Final FSM State", clean_state)
        h4.metric("Anomaly Gate Status", "TRIPPED" if "ALERT" in final_fsm_state or "REPORT" in final_fsm_state else "NORMAL")
        
        # CSV Data Download & Raw Trace Expander
        st.markdown("---")
        d_col1, d_col2 = st.columns([2, 3])
        
        df_trace = pd.DataFrame({
            "Cycle": hw_cycles,
            "Power (I²+Q²)": hw_powers,
            "Threshold Hit": hw_hits,
            "Counter Value": hw_counts,
            "FSM State": hw_states
        })
        
        with d_col1:
            st.markdown("##### 📥 Export Hardware Simulation Trace")
            csv_data = df_trace.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Cycle Telemetry (CSV)",
                data=csv_data,
                file_name="verilog_telemetry_trace.csv",
                mime="text/csv"
            )
            
        with d_col2:
            st.markdown("##### 💻 Run on EDA Playground / EPWave")
            st.caption("Copy `testbench/eda_playground_top.v` into [edaplayground.com](https://www.edaplayground.com/) with Icarus Verilog to inspect full .vcd waveform timing.")
        
        with st.expander("🔍 Inspect Cycle-by-Cycle Register Telemetry (First 50 Cycles)"):
            st.dataframe(df_trace.head(50), use_container_width=True)

else:
    st.warning("⚠️ No signal data loaded. Please select or synthesize an IQ signal from the sidebar.")
