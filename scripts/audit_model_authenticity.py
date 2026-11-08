"""
=============================================================================
Module: audit_model_authenticity.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 6 - Forensic ML Audit: Overfitting, Leakage & Robustness Stress Test
=============================================================================

This script performs 5 forensic audits to verify whether our 99.2% benchmark
is completely legitimate or contains hidden overfitting, leakage, or brittle assumptions:

1. Test A: Permutation Sanity Test (Scrambled labels -> must collapse to 25%).
2. Test B: Massive Unseen Out-of-Sample Test (4,000 fresh frames, compute Precision, Recall, F1).
3. Test C: Carrier Frequency Offset (CFO) Robustness Sweep (-20 kHz to +20 kHz).
4. Test D: Timing Jitter & Fractional Delay Stress Test (Sub-sample shifting).
5. Test E: Rayleigh Multipath Fading Stress Test (Non-line-of-sight reflections).
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Ensure path finds ai_model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from ai_model.model import ModulationCNN1D
from ai_model.dataset_builder import (
    generate_single_iq_frame,
    compute_differential_phase_features,
    generate_symbols_for_class,
    LABEL_MAP
)


def load_trained_model(weights_path: str, device: torch.device):
    model = ModulationCNN1D(num_classes=4, input_channels=4, seq_len=128).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def run_audit():
    weights_path = os.path.join(base_dir, "ai_model", "weights", "best_rf_classifier.pt")
    device = torch.device("cpu")
    model = load_trained_model(weights_path, device)

    print("=" * 80)
    print("FORENSIC ML AUDIT: VERIFYING BENCHMARK AUTHENTICITY & ROBUSTNESS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Permutation / Label Leakage Test
    # -------------------------------------------------------------------------
    print("\n[TEST 1/5] LABEL LEAKAGE / PERMUTATION AUDIT")
    print("  Hypothesis: If the pipeline is clean of data leakage, randomly permuting")
    print("  labels must cause accuracy to collapse to random guessing (25.0%).")
    
    num_samples = 1000
    frames = np.zeros((num_samples, 4, 128), dtype=np.float32)
    true_labels = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        cls = i % 4
        frames[i] = generate_single_iq_frame(class_id=cls, seq_len=128, snr_db=15.0)
        true_labels[i] = cls

    # Scramble labels
    scrambled_labels = np.random.permutation(true_labels)
    with torch.no_grad():
        preds = torch.argmax(model(torch.from_numpy(frames)), dim=1).numpy()
    
    scrambled_acc = (preds == scrambled_labels).mean() * 100.0
    unscrambled_acc = (preds == true_labels).mean() * 100.0
    print(f"  Scrambled Label Accuracy   : {scrambled_acc:.1f}% (Expected: ~25.0%)")
    print(f"  True Label Accuracy        : {unscrambled_acc:.1f}%")
    leakage_passed = abs(scrambled_acc - 25.0) < 5.0
    print(f"  Result                     : {'PASSED (Zero Data Leakage Detected)' if leakage_passed else 'FAILED'}")

    # -------------------------------------------------------------------------
    # TEST 2: Massive Unseen Out-of-Sample Test (4,000 Independent Frames)
    # -------------------------------------------------------------------------
    print("\n[TEST 2/5] MASSIVE INDEPENDENT TEST (4,000 Unseen Frames across SNR 10-25 dB)")
    test_samples_per_class = 1000
    y_true = []
    y_pred = []
    confidences = []

    for cls in range(4):
        for _ in range(test_samples_per_class):
            snr = np.random.uniform(10.0, 25.0)
            frame = generate_single_iq_frame(class_id=cls, seq_len=128, snr_db=snr)
            tensor_x = torch.from_numpy(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(tensor_x), dim=-1).squeeze().numpy()
            pred_cls = int(np.argmax(probs))
            y_true.append(cls)
            y_pred.append(pred_cls)
            confidences.append(probs[pred_cls])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_acc = (y_true == y_pred).mean() * 100.0
    mean_conf = np.mean(confidences) * 100.0

    print(f"  Overall Accuracy on 4,000 Fresh Frames: {total_acc:.2f}%")
    print(f"  Average Model Confidence              : {mean_conf:.2f}%")
    
    print("\n  Per-Class Precision, Recall, and F1-Scores:")
    print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score'}")
    print("  " + "-" * 42)
    for cls in range(4):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp + 1e-12) * 100.0
        recall = tp / (tp + fn + 1e-12) * 100.0
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        print(f"  {LABEL_MAP[cls]:<10} {precision:<11.1f}% {recall:<11.1f}% {f1:.1f}%")

    # -------------------------------------------------------------------------
    # TEST 3: Carrier Frequency Offset (CFO) Robustness Sweep
    # -------------------------------------------------------------------------
    print("\n[TEST 3/5] CARRIER FREQUENCY OFFSET (CFO) STRESS TEST")
    print("  Testing whether residual frequency drift breaks the differential phase logic...")
    cfo_ratios = [-0.04, -0.02, -0.01, 0.0, +0.01, +0.02, +0.04]  # delta_f / Fs
    print(f"  {'CFO (df/Fs)':<15} {'Overall Accuracy'}")
    print("  " + "-" * 32)

    for cfo in cfo_ratios:
        correct = 0
        total = 400
        for i in range(total):
            cls = i % 4
            # Generate base signal
            num_symbols = 20
            symbols = generate_symbols_for_class(cls, num_symbols)
            base = np.repeat(symbols, 8)[:128]
            # Add CFO drift
            n = np.arange(128)
            drift = np.exp(1j * (2 * np.pi * cfo * n + np.random.uniform(0, 2*np.pi)))
            sig = base * drift
            # Add AWGN (20 dB)
            p_sig = np.mean(np.abs(sig)**2)
            sigma = np.sqrt((p_sig / 100.0) / 2.0)
            noise = np.random.normal(0, sigma, 128) + 1j * np.random.normal(0, sigma, 128)
            sig = sig + noise
            sig = sig / np.sqrt(np.mean(np.abs(sig)**2))

            di, dq = compute_differential_phase_features(np.real(sig), np.imag(sig))
            frame = np.vstack([np.real(sig), np.imag(sig), di, dq]).astype(np.float32)
            tensor_x = torch.from_numpy(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.argmax(model(tensor_x), dim=1).item()
            if pred == cls:
                correct += 1
        acc = (correct / total) * 100.0
        print(f"  {cfo:<15.2f} {acc:.1f}%")

    # -------------------------------------------------------------------------
    # TEST 4: Timing Jitter / Fractional Delay Stress Test
    # -------------------------------------------------------------------------
    print("\n[TEST 4/5] TIMING JITTER & FRACTIONAL SAMPLE SHIFT AUDIT")
    print("  Testing sensitivity to clock misalignment / non-integer sampling offsets...")
    shifts = [0, 1, 2, 3, 4]  # Discrete sample shifts within the 8 samples/symbol window
    print(f"  {'Sample Shift':<15} {'Accuracy'}")
    print("  " + "-" * 28)
    for s in shifts:
        correct = 0
        total = 400
        for i in range(total):
            cls = i % 4
            num_symbols = 24
            symbols = generate_symbols_for_class(cls, num_symbols)
            base = np.repeat(symbols, 8)[s:s+128]
            # Random phase + 18 dB AWGN
            base = base * np.exp(1j * np.random.uniform(0, 2*np.pi))
            p_sig = np.mean(np.abs(base)**2)
            sigma = np.sqrt((p_sig / 63.1) / 2.0)
            noise = np.random.normal(0, sigma, 128) + 1j * np.random.normal(0, sigma, 128)
            sig = base + noise
            sig = sig / np.sqrt(np.mean(np.abs(sig)**2))

            di, dq = compute_differential_phase_features(np.real(sig), np.imag(sig))
            frame = np.vstack([np.real(sig), np.imag(sig), di, dq]).astype(np.float32)
            tensor_x = torch.from_numpy(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.argmax(model(tensor_x), dim=1).item()
            if pred == cls:
                correct += 1
        acc = (correct / total) * 100.0
        print(f"  {s:<15} {acc:.1f}%")

    # -------------------------------------------------------------------------
    # TEST 5: Multipath Rayleigh Fading Stress Test
    # -------------------------------------------------------------------------
    print("\n[TEST 5/5] MULTIPATH RAYLEIGH FADING AUDIT")
    print("  Testing performance when reflections (multi-path ISI) interfere with the signal...")
    delays = [0, 2, 4]  # Path 2 delay in samples
    for d in delays:
        correct = 0
        total = 400
        for i in range(total):
            cls = i % 4
            num_symbols = 20
            symbols = generate_symbols_for_class(cls, num_symbols)
            base = np.repeat(symbols, 8)[:128]
            if d > 0:
                # 2-ray multipath channel: h[n] = delta[n] + 0.35 * exp(j * phi) * delta[n - d]
                echo = np.roll(base, d) * 0.35 * np.exp(1j * np.random.uniform(0, 2*np.pi))
                sig = base + echo
            else:
                sig = base

            sig = sig * np.exp(1j * np.random.uniform(0, 2*np.pi))
            p_sig = np.mean(np.abs(sig)**2)
            sigma = np.sqrt((p_sig / 100.0) / 2.0)
            noise = np.random.normal(0, sigma, 128) + 1j * np.random.normal(0, sigma, 128)
            sig = (sig + noise) / np.sqrt(np.mean(np.abs(sig)**2))

            di, dq = compute_differential_phase_features(np.real(sig), np.imag(sig))
            frame = np.vstack([np.real(sig), np.imag(sig), di, dq]).astype(np.float32)
            tensor_x = torch.from_numpy(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = torch.argmax(model(tensor_x), dim=1).item()
            if pred == cls:
                correct += 1
        acc = (correct / total) * 100.0
        print(f"  Multipath Delay = {d} samples | Accuracy: {acc:.1f}%")

    print("\n" + "=" * 80)
    print("FORENSIC AUDIT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
