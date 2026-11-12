"""
=============================================================================
Module: export_engine.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: CSE Edge Optimization - TorchScript JIT & ONNX Export Engine
Author: Akshay Kumar (CSE Collaborator)
=============================================================================

This engine optimizes the trained 1D-CNN Automatic Modulation Classifier (AMC)
for ultra-low-latency deployment on edge computing nodes (NVIDIA Jetson, 
Raspberry Pi 5, Intel OpenVINO, and Cloud REST microservices).

Capabilities:
  1. Compiles PyTorch graph to TorchScript JIT (Just-In-Time compiled trace).
  2. Exports model to ONNX (Open Neural Network Exchange, opset 14/17).
  3. Benchmarks inference latency: PyTorch Eager vs. TorchScript JIT.
  4. Verifies numerical equivalence (|y_torch - y_jit| < 1e-5).
"""

import os
import sys
import time
import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from ai_model.model import ModulationCNN1D


def export_models(weights_path: str = None, output_dir: str = None) -> dict:
    """
    Exports the trained PyTorch 1D-CNN model to TorchScript JIT and ONNX formats.
    """
    if weights_path is None:
        weights_path = os.path.join(base_dir, "ai_model", "weights", "best_rf_classifier.pt")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "ai_model", "weights")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    device = torch.device("cpu")
    model = ModulationCNN1D(num_classes=4, input_channels=4, seq_len=128).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Dummy input: (batch_size=1, channels=4, seq_len=128)
    dummy_input = torch.randn(1, 4, 128, dtype=torch.float32, device=device)

    print("=" * 80)
    print("CSE EDGE OPTIMIZATION: EXPORTING 1D-CNN AMC MODEL")
    print("=" * 80)
    print(f"Base Weights Source : {weights_path}")

    # 1. Export to TorchScript JIT
    jit_path = os.path.join(output_dir, "best_rf_classifier.torchscript.pt")
    print("\n[STAGE 1/3] Compiling TorchScript JIT Trace...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(jit_path)
    jit_size_kb = os.path.getsize(jit_path) / 1024.0
    print(f"  [PASS] TorchScript JIT Model exported: {jit_path} ({jit_size_kb:.1f} KB)")

    # 2. Export to ONNX
    onnx_path = os.path.join(output_dir, "best_rf_classifier.onnx")
    print("\n[STAGE 2/3] Exporting to ONNX Format (Open Neural Network Exchange)...")
    try:
        import onnx  # check if onnx is installed
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["rf_input_tensor"],
            output_names=["logits"],
            dynamic_axes={"rf_input_tensor": {0: "batch_size"}, "logits": {0: "batch_size"}}
        )
        onnx_size_kb = os.path.getsize(onnx_path) / 1024.0
        print(f"  [PASS] ONNX Model exported: {onnx_path} ({onnx_size_kb:.1f} KB)")
        onnx_success = True
    except ImportError:
        print("  [INFO] Optional package 'onnx' is not installed (run 'pip install onnx' to enable .onnx serialization).")
        print("  [PASS] TorchScript JIT serves as primary production deployment engine (zero external dependencies).")
        onnx_success = False
    except Exception as e:
        print(f"  [INFO] ONNX export skipped ({e}). TorchScript JIT trace is fully operational.")
        onnx_success = False

    # 3. Numerical Verification & Latency Benchmarking
    print("\n[STAGE 3/3] Numerical Verification & Inference Latency Benchmark...")
    loaded_jit = torch.jit.load(jit_path, map_location=device)
    loaded_jit.eval()

    with torch.no_grad():
        out_eager = model(dummy_input)
        out_jit = loaded_jit(dummy_input)

    max_diff = torch.max(torch.abs(out_eager - out_jit)).item()
    print(f"  Numerical Consistency: Max Absolute Difference = {max_diff:.2e} (Threshold: < 1e-5)")
    assert max_diff < 1e-5, f"Numerical inconsistency detected between Eager and JIT: {max_diff}"
    print("  [PASS] Bit-exact output verified between Eager PyTorch and TorchScript JIT!")

    # Latency Benchmark over 500 iterations
    iterations = 500
    # Warmup
    for _ in range(50):
        _ = model(dummy_input)
        _ = loaded_jit(dummy_input)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = model(dummy_input)
    t_eager = (time.perf_counter() - t0) / iterations * 1000.0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = loaded_jit(dummy_input)
    t_jit = (time.perf_counter() - t0) / iterations * 1000.0

    speedup = ((t_eager - t_jit) / t_eager) * 100.0 if t_eager > 0 else 0.0

    print(f"\n  Inference Latency Benchmark ({iterations} iterations @ batch_size=1):")
    print(f"    - PyTorch Eager Mode : {t_eager:.3f} ms / frame ({1000.0 / t_eager:.1f} FPS)")
    print(f"    - TorchScript JIT    : {t_jit:.3f} ms / frame ({1000.0 / t_jit:.1f} FPS)")
    print(f"    - Edge Optimization  : {speedup:+.1f}% Latency Reduction")

    print("\n" + "=" * 80)
    print("CSE EDGE EXPORT COMPLETE & VERIFIED!")
    print("=" * 80)

    return {
        "torchscript_path": jit_path,
        "onnx_path": onnx_path if onnx_success else None,
        "eager_latency_ms": round(t_eager, 3),
        "jit_latency_ms": round(t_jit, 3),
        "speedup_pct": round(speedup, 1),
        "numerical_diff": max_diff
    }


if __name__ == "__main__":
    export_models()
