"""
=============================================================================
Module: fft_analysis.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 3 - Python Digital Signal Processing (DSP): FFT & Spectrum Occupancy
=============================================================================

Theoretical Background:
-----------------------
1. Fast Fourier Transform (FFT):
   Converts discrete time-domain IQ samples x[n] into the frequency domain X[k]:
       X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * k * n / N)
   
   Because our input is complex (I + jQ), the spectrum covers the two-sided
   bandwidth [-Fs/2, +Fs/2]. We use np.fft.fftshift to center 0 Hz (DC) in the 
   middle of the array.

2. Power Spectral Density (PSD) in decibels (dB):
   Linear Power = |X[k]|^2 / (N * Fs)
   Logarithmic PSD (dB) = 10 * log10(Linear Power + epsilon)

3. Spectrum Occupancy (%):
   Divides the monitored bandwidth into discrete frequency bins.
   A bin is considered 'occupied' if its power exceeds an energy threshold P_th:
       Occupancy (%) = (Number of bins where PSD > P_th / Total Bins) * 100
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal


def load_iq_from_csv(filepath: str) -> tuple[np.ndarray, str]:
    """
    Loads complex IQ signal samples from a CSV file.
    
    Parameters:
        filepath : Path to the CSV file (columns: sample_index, I, Q, magnitude, power, label).
        
    Returns:
        tuple : (complex_signal_array, label_str)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
    i_list = []
    q_list = []
    label = "UNKNOWN"
    
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i_list.append(float(row["I"]))
            q_list.append(float(row["Q"]))
            if "label" in row:
                label = row["label"]
                
    complex_signal = np.array(i_list, dtype=np.float64) + 1j * np.array(q_list, dtype=np.float64)
    return complex_signal, label


def compute_fft(signal: np.ndarray, sample_rate: float = 1e6, 
                window: str = "hann") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2-sided Fast Fourier Transform (FFT) and Power Spectral Density (PSD) in dB.
    
    Steps:
      1. Apply a window function (Hann/Blackman) to suppress spectral leakage sidelobes.
      2. Compute the complex FFT using np.fft.fft.
      3. Shift zero-frequency component to center using np.fft.fftshift.
      4. Convert power to logarithmic decibel scale: 10 * log10(power).
      
    Parameters:
        signal      : 1D complex numpy array of IQ samples.
        sample_rate : Sampling frequency (Fs) in Hz (e.g. 1.0 MHz).
        window      : Window type ('hann', 'hamming', 'blackman', or 'rect').
        
    Returns:
        tuple : (freq_bins_hz, psd_db, fft_complex_shifted)
    """
    num_samples = len(signal)
    
    # 1. Apply Window Function to reduce FFT edge discontinuity
    if window == "hann":
        win = np.hanning(num_samples)
    elif window == "hamming":
        win = np.hamming(num_samples)
    elif window == "blackman":
        win = np.blackman(num_samples)
    else:
        win = np.ones(num_samples)
        
    windowed_signal = signal * win
    # Coherent gain compensation for the window
    scale_factor = np.sum(win ** 2)
    
    # 2. Compute Complex FFT
    fft_raw = np.fft.fft(windowed_signal, n=num_samples)
    
    # 3. FFT Shift: arranges frequencies from [-Fs/2 to +Fs/2]
    fft_shifted = np.fft.fftshift(fft_raw)
    freqs = np.fft.fftshift(np.fft.fftfreq(num_samples, d=1.0 / sample_rate))
    
    # 4. Compute Power Spectral Density in dB
    # Avoid log(0) with small epsilon
    eps = 1e-12
    power_linear = (np.abs(fft_shifted) ** 2) / (scale_factor + eps)
    psd_db = 10.0 * np.log10(power_linear + eps)
    
    # Normalize peak to 0 dBFS for intuitive thresholding
    psd_db_normalized = psd_db - np.max(psd_db)
    
    return freqs, psd_db_normalized, fft_shifted


def compute_spectrum_occupancy(psd_db: np.ndarray, 
                               threshold_db: float = None, 
                               adaptive_noise_margin_db: float = 12.0) -> tuple[float, np.ndarray, float]:
    """
    Calculates the percentage of frequency bins currently occupied by transmissions.
    
    Formula:
        Occupancy (%) = (Occupied Bins / Total Bins) * 100
        
    Adaptive Thresholding:
        If threshold_db is None, automatically estimates the noise floor using 
        the median power across all bins and sets:
            Threshold = Median_Noise_Floor + Margin_dB
            
    Parameters:
        psd_db                   : Normalized PSD array in dB.
        threshold_db             : Fixed threshold in dB (optional).
        adaptive_noise_margin_db : Margin above noise floor in dB if threshold_db is None.
        
    Returns:
        tuple : (occupancy_percentage, occupied_boolean_mask, effective_threshold_db)
    """
    total_bins = len(psd_db)
    
    if threshold_db is None:
        # Robust noise floor estimation using median (immune to narrow signal peaks)
        noise_floor_est = np.median(psd_db)
        effective_threshold = noise_floor_est + adaptive_noise_margin_db
    else:
        effective_threshold = threshold_db
        
    # Boolean mask: True where power exceeds threshold
    occupied_mask = psd_db > effective_threshold
    occupied_bins = np.count_nonzero(occupied_mask)
    
    occupancy_pct = (occupied_bins / total_bins) * 100.0
    return occupancy_pct, occupied_mask, effective_threshold


def plot_power_spectrum(freqs: np.ndarray, psd_db: np.ndarray, 
                        threshold_db: float, occupancy_pct: float, 
                        title: str, output_png_path: str):
    """
    Renders an interactive-grade power spectrum graph with threshold line and occupancy badge.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    
    plt.figure(figsize=(11, 5))
    freqs_khz = freqs / 1e3  # Convert to kHz for readability
    
    # Power curve
    plt.plot(freqs_khz, psd_db, color="#0284C7", linewidth=1.4, label="Power Spectral Density (PSD)")
    
    # Threshold Line
    plt.axhline(threshold_db, color="#EF4444", linestyle="--", linewidth=1.5, 
                label=f"Detection Threshold ({threshold_db:.1f} dBFS)")
    
    # Shading occupied regions
    plt.fill_between(freqs_khz, psd_db, threshold_db, 
                     where=(psd_db > threshold_db), color="#EF4444", alpha=0.25, 
                     label="Occupied Spectrum Bins")
    
    # Annotations & styling
    plt.title(f"{title} | Occupancy: {occupancy_pct:.1f}%", fontsize=12, fontweight="bold")
    plt.xlabel("Baseband Frequency [kHz]", fontsize=10)
    plt.ylabel("Normalized Power [dBFS]", fontsize=10)
    plt.ylim(-65, 5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
    print(f"Spectrum plot saved: {output_png_path}")


# =============================================================================
# Standalone Step 3 Verification & Batch DSP Run
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Starting Step 3: Python DSP Pipeline (FFT & Spectrum Occupancy)")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets", "simulated")
    plots_dir = os.path.join(base_dir, "results", "plots")
    
    test_files = [
        ("clean_qpsk.csv", "Clean QPSK Carrier Spectrum", "fft_qpsk_clean.png"),
        ("noisy_bpsk_snr6db.csv", "Noisy BPSK Carrier Spectrum (SNR=6dB)", "fft_bpsk_noisy.png"),
        ("clean_16qam.csv", "16-QAM Signal Spectrum", "fft_16qam.png"),
        ("burst_interference.csv", "QPSK Spectrum with High-Power Burst", "fft_burst_interference.png"),
        ("swept_chirp_jammed.csv", "Wideband Swept-Chirp Jammed Spectrum", "fft_chirp_jammed.png")
    ]
    
    results_summary = []
    
    for filename, title, plot_name in test_files:
        csv_path = os.path.join(datasets_dir, filename)
        if not os.path.exists(csv_path):
            print(f"Skipping {filename} (not found)")
            continue
            
        print(f"\nProcessing DSP for: {filename} ...")
        # 1. Load IQ data
        signal, label = load_iq_from_csv(csv_path)
        
        # 2. Compute 1024-pt FFT & Normalized PSD
        freqs, psd_db, _ = compute_fft(signal, sample_rate=1e6, window="hann")
        
        # 3. Compute Dynamic Spectrum Occupancy
        occupancy_pct, mask, th_used = compute_spectrum_occupancy(psd_db, threshold_db=-20.0)
        
        # 4. Generate Plot
        out_plot = os.path.join(plots_dir, plot_name)
        plot_power_spectrum(freqs, psd_db, threshold_db=th_used, 
                            occupancy_pct=occupancy_pct, title=title, 
                            output_png_path=out_plot)
        
        print(f"  -> Total Bins   : {len(freqs)}")
        print(f"  -> Peak Power   : {np.max(psd_db):.2f} dBFS")
        print(f"  -> Threshold    : {th_used:.1f} dBFS")
        print(f"  -> Occupancy    : {occupancy_pct:.2f}%")
        
        results_summary.append({
            "File": filename,
            "Label": label,
            "Occupancy_%": f"{occupancy_pct:.1f}%",
            "Threshold_dB": f"{th_used:.1f}"
        })
        
    print("\n" + "="*70)
    print("STEP 3 (Part 1/2: FFT & Occupancy) SUMMARY RESULTS:")
    print(f"{'Dataset File':<25} {'Modulation Label':<18} {'Threshold':<12} {'Occupancy'}")
    print("-" * 70)
    for r in results_summary:
        print(f"{r['File']:<25} {r['Label']:<18} {r['Threshold_dB']:<12} {r['Occupancy_%']}")
    print("="*70)
