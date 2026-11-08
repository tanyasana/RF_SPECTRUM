"""
=============================================================================
Module: spectrogram.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 3 - Python Digital Signal Processing (DSP): Spectrogram Waterfall
=============================================================================

What is a Spectrogram?
----------------------
A standard FFT collapses the entire time duration into a single frequency 
spectrum, losing the information of *when* a frequency occurred.

A Spectrogram (computed via Short-Time Fourier Transform, STFT):
1. Divides the IQ sample stream into small overlapping time segments (e.g., 128 samples).
2. Applies a window (e.g. Hann) to each segment.
3. Computes the FFT of each segment.
4. Stacks the FFT magnitudes vertically into a 2D matrix:
     - X-axis : Frequency (kHz)
     - Y-axis : Time (milliseconds or sample block index)
     - Z-axis / Color Intensity : Power (dBFS)

Why is it Critical for RF Interference Detection?
-------------------------------------------------
1. Swept-Frequency Chirp Jammers appear as clear diagonal lines traversing channels.
2. Hopping Transmitters appear as intermittent blocks hopping across frequencies.
3. Pulsed Interference appears as horizontal bright lines at specific timestamps.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

from python.fft_analysis import load_iq_from_csv


def compute_spectrogram(signal: np.ndarray, 
                        sample_rate: float = 1e6, 
                        nperseg: int = 128, 
                        noverlap: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 2D Short-Time Fourier Transform (STFT) Spectrogram of a complex IQ signal.
    
    Parameters:
        signal      : Complex IQ numpy array.
        sample_rate : Sampling rate in Hz.
        nperseg     : Length of each FFT segment (time resolution vs frequency resolution tradeoff).
        noverlap    : Number of overlapping samples between adjacent segments.
        
    Returns:
        tuple : (frequencies_shifted, time_bins, psd_db_2d_shifted)
    """
    # Use scipy.signal.spectrogram with return_onesided=False for complex baseband
    freqs, times, Sxx = sp_signal.spectrogram(
        signal, 
        fs=sample_rate, 
        window="hann", 
        nperseg=nperseg, 
        noverlap=noverlap, 
        return_onesided=False, 
        mode="psd"
    )
    
    # Shift zero-frequency to center of array
    freqs_shifted = np.fft.fftshift(freqs)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    
    # Convert linear power to dBFS
    eps = 1e-12
    psd_db = 10.0 * np.log10(Sxx_shifted + eps)
    psd_db_normalized = psd_db - np.max(psd_db)
    
    return freqs_shifted, times, psd_db_normalized


def plot_spectrogram(freqs: np.ndarray, times: np.ndarray, psd_db: np.ndarray, 
                     title: str, output_png_path: str):
    """
    Renders a 2D Waterfall Spectrogram with colorbar and frequency/time annotations.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    
    plt.figure(figsize=(11, 5.5))
    
    freqs_khz = freqs / 1e3
    times_ms = times * 1e3
    
    # Meshgrid extent for imshow: [x_min, x_max, y_min, y_max]
    extent = [freqs_khz[0], freqs_khz[-1], times_ms[-1], times_ms[0]]
    
    # Render 2D Waterfall heatmap (Viridis / Plasma colormap)
    im = plt.imshow(psd_db.T, aspect="auto", extent=extent, cmap="plasma", 
                    vmin=-45, vmax=0, interpolation="bilinear")
    
    cbar = plt.colorbar(im)
    cbar.set_label("Relative Power [dBFS]", fontsize=10, rotation=270, labelpad=15)
    
    plt.title(f"{title} (Waterfall Spectrogram)", fontsize=12, fontweight="bold")
    plt.xlabel("Frequency [kHz]", fontsize=10)
    plt.ylabel("Time [ms]", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.3, color="white")
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
    print(f"Spectrogram waterfall saved: {output_png_path}")


# =============================================================================
# Standalone Step 3 (Part 2) Verification Runner
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Starting Step 3: Spectrogram Waterfall Processing")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets", "simulated")
    plots_dir = os.path.join(base_dir, "results", "plots")
    
    files_to_process = [
        ("clean_qpsk.csv", "QPSK Normal Transmission", "spectrogram_qpsk.png"),
        ("burst_interference.csv", "QPSK with High-Power Jammer Burst", "spectrogram_burst.png"),
        ("swept_chirp_jammed.csv", "Swept-Frequency Chirp Jammer (Diagonal Signature)", "spectrogram_chirp.png")
    ]
    
    for filename, title, plot_name in files_to_process:
        csv_path = os.path.join(datasets_dir, filename)
        if not os.path.exists(csv_path):
            continue
            
        signal, _ = load_iq_from_csv(csv_path)
        freqs, times, psd_db = compute_spectrogram(signal, sample_rate=1e6, nperseg=128, noverlap=64)
        
        out_png = os.path.join(plots_dir, plot_name)
        plot_spectrogram(freqs, times, psd_db, title=title, output_png_path=out_png)
        print(f"  -> Spectrogram shape : {psd_db.shape} (Freq Bins: {len(freqs)}, Time Bins: {len(times)})")
        
    print("\n" + "="*70)
    print("STEP 3 COMPLETE: FFT, Welch PSD, Occupancy, and Spectrograms generated!")
    print("="*70)
