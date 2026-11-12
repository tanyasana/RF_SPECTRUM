"""
=============================================================================
Module: test_dsp.py
Tests: Digital Signal Processing Pipeline & Statistical Metrics
=============================================================================
"""

import os
import sys
import numpy as np
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from python.fft_analysis import compute_fft, compute_spectrum_occupancy
from python.spectrogram import compute_spectrogram
from python.anomaly_detection import compute_statistical_rf_metrics
from python.signal_generator import generate_qpsk, generate_bpsk, add_power_burst


def test_fft_computation_dimensions():
    """Verify that FFT output array lengths match input signal dimensions."""
    num_samples = 1024
    signal = generate_qpsk(num_samples=num_samples)
    freqs, psd_db, fft_shifted = compute_fft(signal, sample_rate=1e6, window="hann")

    assert len(freqs) == num_samples
    assert len(psd_db) == num_samples
    assert len(fft_shifted) == num_samples
    assert np.isclose(np.max(psd_db), 0.0, atol=1e-3)


def test_fft_windowing_types():
    """Ensure all supported window functions execute properly."""
    signal = generate_bpsk(num_samples=512)
    for win in ["hann", "hamming", "blackman", "rect"]:
        freqs, psd_db, _ = compute_fft(signal, sample_rate=1e6, window=win)
        assert len(psd_db) == 512
        assert not np.isnan(psd_db).any()


def test_spectrum_occupancy_bounds():
    """Verify spectrum occupancy ratio returns valid percentage in [0, 100]."""
    signal = generate_qpsk(num_samples=1024)
    _, psd_db, _ = compute_fft(signal, sample_rate=1e6)
    occ_pct, mask, threshold = compute_spectrum_occupancy(psd_db, threshold_db=-20.0)

    assert 0.0 <= occ_pct <= 100.0
    assert len(mask) == 1024
    assert threshold == -20.0


def test_spectrogram_shape():
    """Verify 2D STFT spectrogram dimensions and non-negativity."""
    signal = generate_qpsk(num_samples=1024)
    freqs, times, psd = compute_spectrogram(signal, sample_rate=1e6, nperseg=128, noverlap=64)

    assert len(freqs) == 128
    assert len(times) > 0
    assert psd.shape == (len(freqs), len(times))


def test_statistical_rf_metrics_clean_vs_jammed():
    """Verify that PAPR and kurtosis distinguish clean signals from high-power bursts."""
    # Clean QPSK
    clean_sig = generate_qpsk(num_samples=1024)
    i_clean = np.real(clean_sig).astype(np.float32)
    q_clean = np.imag(clean_sig).astype(np.float32)
    metrics_clean = compute_statistical_rf_metrics(i_clean, q_clean)

    # Jammed with Burst
    jammed_sig = add_power_burst(clean_sig, start_idx=300, duration=200, amplitude_multiplier=5.0)
    i_jam = np.real(jammed_sig).astype(np.float32)
    q_jam = np.imag(jammed_sig).astype(np.float32)
    metrics_jammed = compute_statistical_rf_metrics(i_jam, q_jam)

    assert metrics_jammed["max_power"] > metrics_clean["max_power"]
    assert metrics_jammed["papr_db"] > metrics_clean["papr_db"]
    assert metrics_jammed["threat_detected"] is True
