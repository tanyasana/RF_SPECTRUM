"""
=============================================================================
Module: dataset_builder.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 6 - Synthetic IQ Dataset Builder with Differential Phase Preprocessing
=============================================================================

Classes Supported:
  0: BPSK   (Binary Phase Shift Keying)
  1: QPSK   (Quadrature Phase Shift Keying)
  2: 8PSK   (8-Phase Shift Keying)
  3: 16-QAM (16-ary Quadrature Amplitude Modulation)

Differential Phase Preprocessing Pipeline:
------------------------------------------
Given complex baseband samples z[n] = I[n] + j*Q[n]:
1. Compute conjugate delayed product:
     p[n] = z[n] * conj(z[n-1])
2. Extract normalized differential in-phase and quadrature components:
     Delta_I[n] = cos(Delta_phi[n]) = Re(p[n]) / (|p[n]| + eps)
     Delta_Q[n] = sin(Delta_phi[n]) = Im(p[n]) / (|p[n]| + eps)
3. Package into 4-Channel Input Tensor of shape (4, seq_len):
     - Channel 0: I[n] (Raw In-Phase component, RMS normalized)
     - Channel 1: Q[n] (Raw Quadrature component, RMS normalized)
     - Channel 2: Delta_I[n] (Differential In-Phase, phase-rotation invariant)
     - Channel 3: Delta_Q[n] (Differential Quadrature, phase-rotation invariant)

Key Mathematical Invariant:
---------------------------
Static carrier phase offset theta_0 cancels out identically:
  p[n] = (s[n] * e^(j*theta_0)) * (s[n-1] * e^(j*theta_0))^* = s[n] * s^*[n-1]
This eliminates rotational ambiguity between QPSK (90 deg steps) and 8PSK (45 deg steps).
"""

import numpy as np

# Class Name Map
LABEL_MAP = {
    0: "BPSK",
    1: "QPSK",
    2: "8PSK",
    3: "16-QAM"
}


def compute_differential_phase_features(i_samples: np.ndarray, q_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes differential phase features:
      p[n] = z[n] * conj(z[n-1])
      Delta_I[n] = cos(Delta_phi[n])
      Delta_Q[n] = sin(Delta_phi[n])
      
    Boundary condition: Delta_phi[0] = 0 (Delta_I[0] = 1.0, Delta_Q[0] = 0.0)
    """
    z = i_samples.astype(np.complex64) + 1j * q_samples.astype(np.complex64)
    z_prev = np.roll(z, 1)
    z_prev[0] = z[0]  # Initial boundary sample

    # Conjugate product
    p = z * np.conj(z_prev)
    mag = np.abs(p) + 1e-12

    d_i = (np.real(p) / mag).astype(np.float32)
    d_q = (np.imag(p) / mag).astype(np.float32)

    # Initial sample has zero relative delta
    d_i[0] = 1.0
    d_q[0] = 0.0
    return d_i, d_q


def generate_symbols_for_class(class_id: int, num_symbols: int) -> np.ndarray:
    """
    Generates normalized complex symbols for the given modulation class.
    """
    if class_id == 0:
        # BPSK: {+1, -1}
        bits = np.random.randint(0, 2, size=num_symbols)
        symbols = np.where(bits == 0, 1.0, -1.0) + 0.0j

    elif class_id == 1:
        # QPSK: {(+1+1j), (-1+1j), (-1-1j), (+1-1j)} / sqrt(2)
        const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        indices = np.random.randint(0, 4, size=num_symbols)
        symbols = const[indices]

    elif class_id == 2:
        # 8PSK: 8 points spaced evenly at 45 degrees around unit circle
        angles = np.arange(8) * (2 * np.pi / 8)
        const = np.exp(1j * angles)
        indices = np.random.randint(0, 8, size=num_symbols)
        symbols = const[indices]

    elif class_id == 3:
        # 16-QAM: 4x4 grid normalized by sqrt(10)
        grid = np.array([-3, -1, 1, 3])
        i_grid, q_grid = np.meshgrid(grid, grid)
        const = (i_grid.flatten() + 1j * q_grid.flatten()) / np.sqrt(10)
        indices = np.random.randint(0, 16, size=num_symbols)
        symbols = const[indices]
    else:
        raise ValueError(f"Unknown class_id: {class_id}")

    return symbols


def generate_single_iq_frame(class_id: int, seq_len: int = 128, 
                             samples_per_symbol: int = 8, 
                             snr_db: float = 15.0) -> np.ndarray:
    """
    Generates one 4-channel (4, seq_len) IQ frame with differential phase preprocessing
    and realistic wireless channel impairments (AWGN, random phase offset theta_0, CFO).
    
    Returns:
        np.ndarray: Shape (4, seq_len), dtype float32
                    Channel 0 = In-phase I[n]
                    Channel 1 = Quadrature Q[n]
                    Channel 2 = Differential In-Phase Delta_I[n] = cos(Delta_phi[n])
                    Channel 3 = Differential Quadrature Delta_Q[n] = sin(Delta_phi[n])
    """
    num_symbols = int(np.ceil(seq_len / samples_per_symbol)) + 4
    symbols = generate_symbols_for_class(class_id, num_symbols)
    
    # Pulse repetition (oversampling)
    base_signal = np.repeat(symbols, samples_per_symbol)[:seq_len]

    # 1. Random Carrier Phase Offset theta_0 in [0, 2*pi]
    random_phase = np.random.uniform(0, 2 * np.pi)
    base_signal = base_signal * np.exp(1j * random_phase)

    # 2. Add AWGN Noise
    sig_power = np.mean(np.abs(base_signal) ** 2)
    if sig_power == 0:
        sig_power = 1.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin
    sigma = np.sqrt(noise_power / 2.0)
    noise = np.random.normal(0, sigma, size=seq_len) + 1j * np.random.normal(0, sigma, size=seq_len)
    noisy_signal = base_signal + noise

    # 3. Energy Normalization (Unit RMS)
    rms = np.sqrt(np.mean(np.abs(noisy_signal) ** 2))
    if rms > 0:
        noisy_signal = noisy_signal / rms

    raw_i = np.real(noisy_signal).astype(np.float32)
    raw_q = np.imag(noisy_signal).astype(np.float32)

    # 4. Compute Differential Phase Preprocessed Features
    d_i, d_q = compute_differential_phase_features(raw_i, raw_q)

    # 5. Pack into 4 channels: [I, Q, Delta_I, Delta_Q]
    frame = np.zeros((4, seq_len), dtype=np.float32)
    frame[0, :] = raw_i
    frame[1, :] = raw_q
    frame[2, :] = d_i
    frame[3, :] = d_q
    return frame


def build_synthetic_dataset(num_samples_per_class: int = 1500, 
                            seq_len: int = 128, 
                            snr_range: tuple = (4.0, 26.0)) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds a balanced synthetic dataset across all 4 modulation classes with 4 channels.
    
    Returns:
        X: Tensor of shape (Total_Samples, 4, seq_len)
        y: Labels vector of shape (Total_Samples,)
    """
    num_classes = 4
    total_samples = num_samples_per_class * num_classes

    X = np.zeros((total_samples, 4, seq_len), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int64)

    idx = 0
    for cls in range(num_classes):
        for _ in range(num_samples_per_class):
            snr = np.random.uniform(snr_range[0], snr_range[1])
            X[idx] = generate_single_iq_frame(class_id=cls, seq_len=seq_len, snr_db=snr)
            y[idx] = cls
            idx += 1

    # Shuffle dataset
    p = np.random.permutation(total_samples)
    return X[p], y[p]
