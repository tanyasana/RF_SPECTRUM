"""
=============================================================================
Module: signal_generator.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 2 - Generate or Obtain IQ Signal Data
=============================================================================

What is IQ Data?
----------------
In wireless communication, any radio frequency (RF) signal s(t) can be 
decomposed into two orthogonal sinusoids 90 degrees out of phase:
    s(t) = I(t) * cos(2*pi*fc*t) - Q(t) * sin(2*pi*fc*t)

Where:
    - I(t) is the In-Phase component (along the cosine axis / real axis).
    - Q(t) is the Quadrature component (along the sine axis / imaginary axis).

In discrete baseband processing, we represent each sample as a complex number:
    x[n] = I[n] + j * Q[n]

Why is IQ Data Useful?
----------------------
1. Eliminates ambiguity between positive and negative frequencies.
2. Direct calculation of instantaneous power: P = I^2 + Q^2.
3. Direct calculation of instantaneous phase: theta = arctan(Q / I).
4. Enables easy generation of standard digital modulation schemes 
   (BPSK, QPSK, 16-QAM) and realistic channel impairments (noise, jammers).
"""

import os
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def generate_carrier_tone(num_samples: int = 1024, sample_rate: float = 1e6, freq: float = 100e3) -> np.ndarray:
    """
    Generates a pure complex sinusoidal carrier tone:
        x[n] = exp(j * 2 * pi * f * n / Fs)
    
    Parameters:
        num_samples : Total number of discrete time samples.
        sample_rate : Sampling frequency (Fs) in Hz (e.g. 1 MHz).
        freq        : Frequency of the tone in Hz (e.g. 100 kHz).
        
    Returns:
        np.ndarray : Complex 1D array of shape (num_samples,).
    """
    t = np.arange(num_samples) / sample_rate
    # Euler's formula: exp(j * theta) = cos(theta) + j * sin(theta)
    signal = np.exp(1j * 2 * np.pi * freq * t)
    return signal


def generate_bpsk(num_samples: int = 1024, samples_per_symbol: int = 8, snr_db: float = None) -> np.ndarray:
    """
    Generates Binary Phase Shift Keying (BPSK) signal.
    
    Modulation Concept:
        BPSK maps bits (0 or 1) to phases:
        - Bit 0 -> Constellation point: +1 + 0j
        - Bit 1 -> Constellation point: -1 + 0j
        All information is in the I component; Q is ideally 0.
        
    Parameters:
        num_samples        : Total output samples.
        samples_per_symbol : How many samples represent one symbol (oversampling).
        snr_db             : Optional Signal-to-Noise Ratio in dB.
        
    Returns:
        np.ndarray : Complex 1D array of modulated baseband samples.
    """
    num_symbols = int(np.ceil(num_samples / samples_per_symbol))
    # Random bits: 0 or 1
    bits = np.random.randint(0, 2, size=num_symbols)
    # Map: 0 -> +1.0, 1 -> -1.0
    symbols = np.where(bits == 0, 1.0, -1.0) + 0.0j
    
    # Pulse shaping / Oversampling: repeat each symbol for 'samples_per_symbol' samples
    raw_signal = np.repeat(symbols, samples_per_symbol)[:num_samples]
    
    # Add noise if requested
    if snr_db is not None:
        raw_signal = add_awgn_noise(raw_signal, snr_db)
        
    return raw_signal


def generate_qpsk(num_samples: int = 1024, samples_per_symbol: int = 8, snr_db: float = None) -> np.ndarray:
    """
    Generates Quadrature Phase Shift Keying (QPSK) signal.
    
    Modulation Concept:
        QPSK transmits 2 bits per symbol across 4 distinct constellation points:
        - 00 -> (+1 + 1j) / sqrt(2)
        - 01 -> (-1 + 1j) / sqrt(2)
        - 11 -> (-1 - 1j) / sqrt(2)
        - 10 -> (+1 - 1j) / sqrt(2)
        Both I and Q carry independent data bits simultaneously.
        
    Parameters:
        num_samples        : Total output samples.
        samples_per_symbol : Number of samples per symbol.
        snr_db             : Optional SNR in dB.
    """
    num_symbols = int(np.ceil(num_samples / samples_per_symbol))
    # 4 Constellation alphabet points normalized to unit average power
    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    symbol_indices = np.random.randint(0, 4, size=num_symbols)
    symbols = constellation[symbol_indices]
    
    raw_signal = np.repeat(symbols, samples_per_symbol)[:num_samples]
    
    if snr_db is not None:
        raw_signal = add_awgn_noise(raw_signal, snr_db)
        
    return raw_signal


def generate_16qam(num_samples: int = 1024, samples_per_symbol: int = 8, snr_db: float = None) -> np.ndarray:
    """
    Generates 16-ary Quadrature Amplitude Modulation (16-QAM) signal.
    
    Modulation Concept:
        Transmits 4 bits per symbol using 16 constellation points arranged 
        in a 4x4 grid: I in {-3, -1, +1, +3}, Q in {-3, -1, +1, +3}.
        Normalized by sqrt(10) so the average symbol power is 1.0.
    """
    num_symbols = int(np.ceil(num_samples / samples_per_symbol))
    grid = np.array([-3, -1, 1, 3])
    # Create all 16 grid coordinates
    i_vals, q_vals = np.meshgrid(grid, grid)
    constellation = (i_vals.flatten() + 1j * q_vals.flatten()) / np.sqrt(10)
    
    symbol_indices = np.random.randint(0, 16, size=num_symbols)
    symbols = constellation[symbol_indices]
    
    raw_signal = np.repeat(symbols, samples_per_symbol)[:num_samples]
    
    if snr_db is not None:
        raw_signal = add_awgn_noise(raw_signal, snr_db)
        
    return raw_signal


def add_awgn_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to a complex signal.
    
    Physics / Math:
        1. Calculate signal power: P_sig = mean(|signal|^2)
        2. Convert SNR from dB to linear: SNR_lin = 10^(SNR_dB / 10)
        3. Noise power: P_noise = P_sig / SNR_lin
        4. Noise variance per dimension (I and Q each): sigma = sqrt(P_noise / 2)
        5. Noise: n[k] = N(0, sigma^2) + j * N(0, sigma^2)
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power == 0:
        sig_power = 1.0
        
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_sigma = np.sqrt(noise_power / 2.0)
    
    noise_i = np.random.normal(0, noise_sigma, size=len(signal))
    noise_q = np.random.normal(0, noise_sigma, size=len(signal))
    noise = noise_i + 1j * noise_q
    
    return signal + noise


def add_frequency_offset(signal: np.ndarray, offset_freq: float, sample_rate: float) -> np.ndarray:
    """
    Simulates Local Oscillator (LO) frequency drift or Doppler shift:
        x_shifted[n] = x[n] * exp(j * 2 * pi * f_offset * n / Fs)
    """
    n = np.arange(len(signal))
    drift = np.exp(1j * 2 * np.pi * offset_freq * n / sample_rate)
    return signal * drift


def add_power_burst(signal: np.ndarray, start_idx: int = 300, duration: int = 150, amplitude_multiplier: float = 6.0) -> np.ndarray:
    """
    Simulates a sudden high-power burst interference or rogue transmitter spike.
    This creates an intentional anomaly to trigger our Verilog threshold comparator.
    
    Parameters:
        start_idx            : Starting sample index of the interference.
        duration             : Number of samples the burst lasts.
        amplitude_multiplier : Surge factor (e.g., 6x normal amplitude).
    """
    modified = signal.copy()
    end_idx = min(start_idx + duration, len(signal))
    # Multiply the amplitude of the burst segment
    modified[start_idx:end_idx] *= amplitude_multiplier
    return modified


def generate_chirp_jammer(num_samples: int = 1024, sample_rate: float = 1e6, 
                          f_start: float = -300e3, f_end: float = 300e3, 
                          power_multiplier: float = 3.0) -> np.ndarray:
    """
    Simulates an intentional Swept-Frequency Chirp Jammer:
        Frequency ramps linearly from f_start to f_end over the observation window:
        f(t) = f_start + ((f_end - f_start) / T) * t
        Phase(t) = 2*pi * integral(f(t) dt) = 2*pi * (f_start * t + 0.5 * k * t^2)
    """
    t = np.arange(num_samples) / sample_rate
    t_max = num_samples / sample_rate
    k = (f_end - f_start) / t_max
    phase = 2 * np.pi * (f_start * t + 0.5 * k * (t ** 2))
    jammer = power_multiplier * np.exp(1j * phase)
    return jammer


def save_iq_to_csv(filepath: str, signal: np.ndarray, label: str = "NORMAL"):
    """
    Exports complex IQ data into standard CSV format for pipeline interoperability:
    Columns: [sample_index, I, Q, magnitude, power, label]
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "I", "Q", "magnitude", "power", "label"])
        for idx, val in enumerate(signal):
            i_val = float(np.real(val))
            q_val = float(np.imag(val))
            mag = float(np.abs(val))
            pwr = float(i_val**2 + q_val**2)
            writer.writerow([idx, f"{i_val:.6f}", f"{q_val:.6f}", f"{mag:.6f}", f"{pwr:.6f}", label])
    print(f"Saved {len(signal)} samples to: {filepath}")


def plot_iq_constellation_and_time(signal: np.ndarray, title: str, output_png_path: str):
    """
    Plots:
      1. Time domain waveform: I(t) and Q(t) vs Sample Index.
      2. Constellation diagram: Q (Quadrature) vs I (In-phase) scatter plot.
    """
    if not HAS_MATPLOTLIB:
        print(f"[Info] Matplotlib is still finishing installation. Skipping plot: {output_png_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    fig, (ax_time, ax_iq) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Time-Domain Plot (First 150 samples for visual clarity)
    display_n = min(150, len(signal))
    ax_time.plot(np.real(signal[:display_n]), label="In-Phase (I)", color="#0284C7", linewidth=1.5)
    ax_time.plot(np.imag(signal[:display_n]), label="Quadrature (Q)", color="#EF4444", linewidth=1.5, linestyle="--")
    ax_time.set_title("Time Domain: I(t) & Q(t) Waveforms (First 150 samples)", fontsize=11, fontweight="bold")
    ax_time.set_xlabel("Sample Index [n]")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, linestyle=":", alpha=0.6)
    ax_time.legend(loc="upper right")
    
    # 2. Constellation Diagram (Q vs I)
    ax_iq.scatter(np.real(signal), np.imag(signal), alpha=0.6, color="#0E7490", edgecolors="none", s=25)
    ax_iq.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_iq.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_iq.set_title("IQ Constellation (Q vs I Plane)", fontsize=11, fontweight="bold")
    ax_iq.set_xlabel("In-Phase (I)")
    ax_iq.set_ylabel("Quadrature (Q)")
    ax_iq.grid(True, linestyle=":", alpha=0.6)
    
    # Square aspect ratio for true constellation geometry
    max_val = max(1.5, np.max(np.abs(signal)) * 1.1)
    ax_iq.set_xlim(-max_val, max_val)
    ax_iq.set_ylim(-max_val, max_val)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_png_path}")


# =============================================================================
# Standalone Demonstration / Verification
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Starting Step 2: Signal Generation & Dataset Creation")
    print("="*70)
    
    np.random.seed(42)  # For reproducible test cases
    sample_rate = 1e6   # 1 MSPS (Mega Sample Per Second)
    num_samples = 1024  # Standard FFT window size
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets", "simulated")
    plots_dir = os.path.join(base_dir, "results", "plots")
    
    # 1. Clean QPSK Signal
    print("\n[1/5] Generating Clean QPSK Signal...")
    qpsk_clean = generate_qpsk(num_samples=num_samples, samples_per_symbol=8, snr_db=25.0)
    save_iq_to_csv(os.path.join(datasets_dir, "clean_qpsk.csv"), qpsk_clean, label="QPSK_CLEAN")
    plot_iq_constellation_and_time(qpsk_clean, "Clean QPSK Signal (SNR = 25 dB)", 
                                   os.path.join(plots_dir, "iq_qpsk_clean.png"))
    
    # 2. Noisy BPSK Signal (Low SNR = 6 dB)
    print("\n[2/5] Generating Noisy BPSK Signal...")
    bpsk_noisy = generate_bpsk(num_samples=num_samples, samples_per_symbol=8, snr_db=6.0)
    save_iq_to_csv(os.path.join(datasets_dir, "noisy_bpsk_snr6db.csv"), bpsk_noisy, label="BPSK_NOISY")
    plot_iq_constellation_and_time(bpsk_noisy, "Noisy BPSK Signal (SNR = 6 dB)", 
                                   os.path.join(plots_dir, "iq_bpsk_noisy.png"))
    
    # 3. 16-QAM Signal (High-order Modulation)
    print("\n[3/5] Generating 16-QAM Signal...")
    qam16 = generate_16qam(num_samples=num_samples, samples_per_symbol=8, snr_db=20.0)
    save_iq_to_csv(os.path.join(datasets_dir, "clean_16qam.csv"), qam16, label="16QAM")
    plot_iq_constellation_and_time(qam16, "16-QAM Signal (4x4 Constellation Grid, SNR = 20 dB)", 
                                   os.path.join(plots_dir, "iq_16qam.png"))
    
    # 4. Signal with High-Power Interference Burst (Rogue Transmitter)
    print("\n[4/5] Generating Signal with Rogue Interference Burst...")
    burst_signal = add_power_burst(qpsk_clean, start_idx=350, duration=120, amplitude_multiplier=5.0)
    save_iq_to_csv(os.path.join(datasets_dir, "burst_interference.csv"), burst_signal, label="BURST_JAMMER")
    plot_iq_constellation_and_time(burst_signal, "QPSK with High-Power Burst Interference (Samples 350-470)", 
                                   os.path.join(plots_dir, "iq_burst_interference.png"))
    
    # 5. Swept-Frequency Chirp Jammer
    print("\n[5/5] Generating Swept-Frequency Chirp Jammer...")
    chirp = generate_chirp_jammer(num_samples=num_samples, sample_rate=sample_rate, 
                                  f_start=-300e3, f_end=300e3, power_multiplier=3.0)
    # Combine normal QPSK + Chirp jammer
    jammed_signal = qpsk_clean + chirp
    save_iq_to_csv(os.path.join(datasets_dir, "swept_chirp_jammed.csv"), jammed_signal, label="CHIRP_JAMMER")
    plot_iq_constellation_and_time(jammed_signal, "Signal Contaminated by Swept Chirp Jammer", 
                                   os.path.join(plots_dir, "iq_chirp_jammed.png"))
    
    print("\n" + "="*70)
    print("STEP 2 COMPLETE: All test datasets and constellation plots generated!")
    print(f"Datasets: {datasets_dir}")
    print(f"Plots   : {plots_dir}")
    print("="*70)
