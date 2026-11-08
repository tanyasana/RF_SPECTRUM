"""
=============================================================================
Module: classifier.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 6 - Real-Time AI Inference Engine & Hybrid Jamming Detector
=============================================================================

Inference Pipeline with Differential Phase Preprocessing:
---------------------------------------------------------
1. Ingests raw complex IQ samples from CSV or live RF buffer.
2. Slices signal into 128-sample temporal frames.
3. Computes 4-Channel Preprocessed Features for each frame:
     - Ch 0: I[n] (RMS normalized)
     - Ch 1: Q[n] (RMS normalized)
     - Ch 2: Delta_I[n] = cos(Delta_phi[n])
     - Ch 3: Delta_Q[n] = sin(Delta_phi[n])
   where Delta_phi[n] = arg(z[n] * conj(z[n-1])) cancels static carrier phase offset.
4. Executes forward pass on 1D-CNN: ai_model/weights/best_rf_classifier.pt
5. Aggregates softmax class probabilities across frames.
6. Evaluates Shannon Prediction Entropy:
       H = - sum( P_i * log2(P_i) )
7. Hybrid Jamming Gate:
       If Prediction Entropy is abnormally high (ambiguous RF signature)
       OR if Peak Power exceeds anomalous threshold:
           Status = "THREAT / UNKNOWN JAMMER"
       Else:
           Status = "NORMAL TRANSMISSION"
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F

from ai_model.model import ModulationCNN1D
from ai_model.dataset_builder import LABEL_MAP, compute_differential_phase_features


class RFSignalClassifier:
    def __init__(self, weights_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if weights_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(base_dir, "ai_model", "weights", "best_rf_classifier.pt")

        self.weights_path = weights_path
        self.model = ModulationCNN1D(num_classes=4, input_channels=4, seq_len=128).to(self.device)

        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
        else:
            self.is_loaded = False

    def predict_signal(self, i_samples: np.ndarray, q_samples: np.ndarray, 
                       seq_len: int = 128, power_threshold: float = 2.5) -> dict:
        """
        Runs AMC inference across a full signal array with 4-channel differential phase preprocessing.
        
        Parameters:
            i_samples       : 1D array of In-Phase samples.
            q_samples       : 1D array of Quadrature samples.
            seq_len         : Slicing frame length (128).
            power_threshold : Threshold above which signal is considered an energy burst anomaly.
            
        Returns:
            dict: Comprehensive prediction metrics and jamming anomaly flags.
        """
        if not self.is_loaded:
            raise RuntimeError(f"Trained model weights not found at: {self.weights_path}. Run train.py first!")

        num_samples = len(i_samples)
        num_frames = num_samples // seq_len

        if num_frames == 0:
            # Pad if shorter than 128
            padded_i = np.pad(i_samples, (0, seq_len - num_samples), mode="constant")
            padded_q = np.pad(q_samples, (0, seq_len - num_samples), mode="constant")
            rms = np.sqrt(np.mean(padded_i**2 + padded_q**2))
            if rms > 0:
                padded_i = padded_i / rms
                padded_q = padded_q / rms
            d_i, d_q = compute_differential_phase_features(padded_i, padded_q)
            frames = np.expand_dims(np.vstack([padded_i, padded_q, d_i, d_q]), axis=0).astype(np.float32)
        else:
            frames = np.zeros((num_frames, 4, seq_len), dtype=np.float32)
            for idx in range(num_frames):
                start = idx * seq_len
                end = start + seq_len
                sl_i = i_samples[start:end]
                sl_q = q_samples[start:end]

                # RMS normalize each frame
                rms = np.sqrt(np.mean(sl_i**2 + sl_q**2))
                if rms > 0:
                    sl_i = sl_i / rms
                    sl_q = sl_q / rms

                d_i, d_q = compute_differential_phase_features(sl_i, sl_q)

                frames[idx, 0, :] = sl_i
                frames[idx, 1, :] = sl_q
                frames[idx, 2, :] = d_i
                frames[idx, 3, :] = d_q

        # Convert to Tensor & Run Forward Pass
        tensor_in = torch.from_numpy(frames).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        # Mean probability vector across frames
        mean_probs = np.mean(probs, axis=0)
        best_class_id = int(np.argmax(mean_probs))
        confidence = float(mean_probs[best_class_id]) * 100.0

        # Calculate Shannon Entropy: H = - sum(P * log2(P))
        eps = 1e-12
        entropy = float(-np.sum(mean_probs * np.log2(mean_probs + eps)))

        # Power calculation for physical anomaly verification
        avg_power = float(np.mean(i_samples**2 + q_samples**2))
        max_power = float(np.max(i_samples**2 + q_samples**2))

        # Hybrid Jamming Detection Logic:
        is_jamming = False
        anomaly_reason = "Normal Signal Signature"

        if max_power > power_threshold:
            is_jamming = True
            anomaly_reason = f"High-Power Jamming Burst Detected (Max Power: {max_power:.2f} > Limit: {power_threshold:.2f})"
        elif entropy > 1.6:
            is_jamming = True
            anomaly_reason = f"Unrecognized / Noise Jammer Signature (High Prediction Entropy: {entropy:.2f})"

        return {
            "predicted_class": LABEL_MAP[best_class_id],
            "class_id": best_class_id,
            "confidence_pct": round(confidence, 2),
            "probabilities": {LABEL_MAP[i]: round(float(mean_probs[i]) * 100.0, 2) for i in range(4)},
            "shannon_entropy": round(entropy, 3),
            "is_anomaly": is_jamming,
            "anomaly_reason": anomaly_reason,
            "avg_power": round(avg_power, 4),
            "max_power": round(max_power, 4),
            "frames_analyzed": max(1, num_frames)
        }


def classify_csv_file(csv_filepath: str, power_threshold: float = 2.5):
    """
    Convenience function to classify an IQ CSV file and print a diagnostic report.
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"File not found: {csv_filepath}")

    i_list = []
    q_list = []
    with open(csv_filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i_list.append(float(row["I"]))
            q_list.append(float(row["Q"]))

    i_arr = np.array(i_list, dtype=np.float32)
    q_arr = np.array(q_list, dtype=np.float32)

    classifier = RFSignalClassifier()
    res = classifier.predict_signal(i_arr, q_arr, power_threshold=power_threshold)

    print("\n" + "=" * 70)
    print(f"AI MODULATION CLASSIFICATION REPORT: {os.path.basename(csv_filepath)}")
    print("=" * 70)
    print(f"  Predicted Modulation : {res['predicted_class']}")
    print(f"  Confidence Score     : {res['confidence_pct']}%")
    print(f"  Shannon Entropy      : {res['shannon_entropy']} (Ambiguity Metric)")
    print(f"  Class Breakdown      : {res['probabilities']}")
    print(f"  Average Signal Power : {res['avg_power']}")
    print(f"  Maximum Signal Power : {res['max_power']}")
    if res['is_anomaly']:
        print(f"  >>> ALERT STATUS     : ANOMALOUS JAMMER / THREAT DETECTED!")
        print(f"  >>> Diagnosis        : {res['anomaly_reason']}")
    else:
        print(f"  >>> ALERT STATUS     : NORMAL TRANSMISSION (Within Safe Thresholds)")
    print("=" * 70)
    return res


if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_dir, "datasets", "simulated", "clean_qpsk.csv")
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    classify_csv_file(test_file)
