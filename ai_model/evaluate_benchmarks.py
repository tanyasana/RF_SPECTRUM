"""
=============================================================================
Module: evaluate_benchmarks.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 6 - Theoretical vs. Experimental Benchmark Verification
=============================================================================

Differential Phase Augmented Benchmark Verification:
1. Evaluates Classification Accuracy vs. SNR (-6 dB to +24 dB in 3 dB steps)
   with 4-channel differential phase preprocessing.
2. Measures Average Inference Latency (ms per 128-sample frame).
3. Compares Experimental Results with Theoretical Benchmarks:
     - Theoretical Target at High SNR (>= 12 dB): > 95.0%
     - Theoretical Target at Medium SNR (6-9 dB): > 85.0%
     - Theoretical Target at Low SNR (<= -6 dB): 25.0% random floor
4. Plots and exports:
     results/plots/ai_benchmark_accuracy_vs_snr.png
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from ai_model.model import ModulationCNN1D
from ai_model.dataset_builder import generate_single_iq_frame, LABEL_MAP


def evaluate_accuracy_vs_snr(model, device, snr_levels: list, samples_per_snr_per_class: int = 100):
    num_classes = 4
    accuracies = []
    per_class_snr_acc = {cls: [] for cls in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for snr in snr_levels:
            correct_total = 0
            total_samples = 0

            for cls in range(num_classes):
                # Generate 4-channel test frames at this exact SNR
                frames = np.zeros((samples_per_snr_per_class, 4, 128), dtype=np.float32)
                for i in range(samples_per_snr_per_class):
                    frames[i] = generate_single_iq_frame(class_id=cls, seq_len=128, snr_db=snr)

                tensor_x = torch.from_numpy(frames).to(device)
                outputs = model(tensor_x)
                _, preds = torch.max(outputs, 1)

                correct_cls = (preds.cpu().numpy() == cls).sum()
                per_class_snr_acc[cls].append((correct_cls / samples_per_snr_per_class) * 100.0)

                correct_total += correct_cls
                total_samples += samples_per_snr_per_class

            overall_snr_acc = (correct_total / total_samples) * 100.0
            accuracies.append(overall_snr_acc)

    return accuracies, per_class_snr_acc


def measure_inference_latency(model, device, num_trials: int = 200):
    dummy_input = torch.randn(1, 4, 128, dtype=torch.float32).to(device)
    model.eval()

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies_ms = []
    for _ in range(num_trials):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    avg_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    return avg_latency, p95_latency


def plot_benchmark_curve(snr_levels, accuracies, per_class_acc, output_png_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.figure(figsize=(10, 5.5))

    # Overall curve
    plt.plot(snr_levels, accuracies, marker="o", color="#0284C7", linewidth=2.5, 
             label="Overall Average Accuracy", zorder=5)

    # Per-class curves
    colors = ["#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
    for cls, color in zip(range(4), colors):
        plt.plot(snr_levels, per_class_acc[cls], linestyle="--", alpha=0.75, 
                 color=color, label=f"{LABEL_MAP[cls]}")

    # Theoretical Target line (95%)
    plt.axhline(95.0, color="#EF4444", linestyle=":", linewidth=1.5, label="Theoretical High-SNR Benchmark (95%)")
    plt.axhline(25.0, color="#64748B", linestyle=":", linewidth=1.2, label="Random Guessing Floor (25%)")

    plt.title("Modulation Classification Accuracy vs. SNR (Diff Phase Preprocessed)", fontsize=13, fontweight="bold")
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=11)
    plt.ylabel("Classification Accuracy (%)", fontsize=11)
    plt.ylim(10, 102)
    plt.xlim(min(snr_levels) - 1, max(snr_levels) + 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
    print(f"Benchmark plot saved: {output_png_path}")
    sys.stdout.flush()


def run_benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(base_dir, "ai_model", "weights", "best_rf_classifier.pt")
    plot_path = os.path.join(base_dir, "results", "plots", "ai_benchmark_accuracy_vs_snr.png")

    device = torch.device("cpu")
    model = ModulationCNN1D(num_classes=4, input_channels=4, seq_len=128).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("=" * 80)
    print("STEP 6: THEORETICAL VS. EXPERIMENTAL BENCHMARK (DIFF PHASE AUGMENTED)")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Latency Benchmark
    avg_lat, p95_lat = measure_inference_latency(model, device)
    print(f"\n[BENCHMARK 1: INFERENCE LATENCY]")
    print(f"  Theoretical Target : < 5.00 ms per frame")
    print(f"  Experimental Avg   : {avg_lat:.2f} ms")
    print(f"  Experimental P95   : {p95_lat:.2f} ms")
    latency_pass = avg_lat < 5.0
    print(f"  Status             : {'PASS (Meets Real-Time Spec)' if latency_pass else 'FAIL'}")
    sys.stdout.flush()

    # 2. Accuracy vs. SNR Benchmark
    snr_levels = [-6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24]
    print(f"\n[BENCHMARK 2: ACCURACY VS. SNR SPECTRUM]")
    print("Evaluating 400 test frames per SNR step across 11 discrete SNR levels...")
    sys.stdout.flush()
    accuracies, per_class_acc = evaluate_accuracy_vs_snr(model, device, snr_levels)

    print(f"\n{'SNR (dB)':<10} {'Overall Acc':<14} {'BPSK':<10} {'QPSK':<10} {'8PSK':<10} {'16-QAM':<10}")
    print("-" * 65)
    for idx, snr in enumerate(snr_levels):
        print(f"{snr:<10} {accuracies[idx]:<13.1f}% {per_class_acc[0][idx]:<9.1f}% {per_class_acc[1][idx]:<9.1f}% {per_class_acc[2][idx]:<9.1f}% {per_class_acc[3][idx]:<9.1f}%")
    sys.stdout.flush()

    plot_benchmark_curve(snr_levels, accuracies, per_class_acc, plot_path)

    # 3. High-SNR Benchmark Check (>= 12 dB)
    high_snr_indices = [i for i, snr in enumerate(snr_levels) if snr >= 12]
    high_snr_avg = float(np.mean([accuracies[i] for i in high_snr_indices]))
    peak_accuracy = float(np.max(accuracies))

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY:")
    print("=" * 80)
    print(f"  Theoretical High-SNR Goal (>12 dB) : >= 95.0%")
    print(f"  Experimental High-SNR Avg (>12 dB)  : {high_snr_avg:.1f}%")
    print(f"  Experimental Peak Accuracy         : {peak_accuracy:.1f}%")

    matches = high_snr_avg >= 95.0
    if matches:
        print("\n>>> CONCLUSION: THEORETICAL BENCHMARK MATCHED EXPERIMENTAL BENCHMARK!")
    else:
        gap = 95.0 - high_snr_avg
        print(f"\n>>> CONCLUSION: BENCHMARK GAP DETECTED (-{gap:.1f}% below theoretical 95% target).")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    run_benchmark()
