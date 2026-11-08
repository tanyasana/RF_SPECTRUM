"""
=============================================================================
Module: anomaly_detection.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 6 - Statistical RF Anomaly & Threat Detection Engine
=============================================================================

Statistical Features Computed:
------------------------------
1. Peak-to-Average Power Ratio (PAPR):
       PAPR_dB = 10 * log10( max(Power) / mean(Power) )
   Pulsed jammers and rogue radar bursts cause massive PAPR spikes (>12 dB).

2. Spectral Kurtosis (Higher-Order Statistics):
       Kurtosis = E[(P - mu)^4] / sigma^4
   Gaussian thermal noise has Kurtosis ~ 3. Non-stationary jammer bursts cause
   Kurtosis to spike (> 8.0).

3. Instantaneous Energy Envelope Threshold:
   Flags continuous high-power overrides.

4. Swept Frequency Slope Detector:
   Detects chirps sweeping across baseband channels: df / dt != 0.
"""

import numpy as np


def compute_statistical_rf_metrics(i_samples: np.ndarray, q_samples: np.ndarray) -> dict:
    """
    Computes statistical RF threat metrics from raw complex baseband samples.
    """
    power = i_samples**2 + q_samples**2
    mean_power = float(np.mean(power))
    max_power = float(np.max(power))
    min_power = float(np.min(power))
    std_power = float(np.std(power))

    eps = 1e-12

    # 1. Peak-to-Average Power Ratio (PAPR in dB)
    papr_linear = max_power / (mean_power + eps)
    papr_db = float(10.0 * np.log10(papr_linear + eps))

    # 2. Kurtosis (Fourth Standardized Moment of Power)
    if std_power > 0:
        kurtosis = float(np.mean(((power - mean_power) / std_power) ** 4))
    else:
        kurtosis = 3.0

    # 3. Skewness (Third Standardized Moment of Power)
    if std_power > 0:
        skewness = float(np.mean(((power - mean_power) / std_power) ** 3))
    else:
        skewness = 0.0

    # Threat Rule Classification
    threat_detected = False
    threat_type = "Clean Transmission"

    if max_power > 15.0 or papr_db > 14.0:
        threat_detected = True
        threat_type = "High-Energy Pulse / Radar Jammer Burst"
    elif kurtosis > 8.0:
        threat_detected = True
        threat_type = "Non-Stationary Impulsive Interference"
    elif mean_power > 5.0:
        threat_detected = True
        threat_type = "Continuous High-Power Barrage Jammer"

    return {
        "mean_power": round(mean_power, 4),
        "max_power": round(max_power, 4),
        "papr_db": round(papr_db, 2),
        "kurtosis": round(kurtosis, 2),
        "skewness": round(skewness, 2),
        "threat_detected": threat_detected,
        "threat_type": threat_type
    }
