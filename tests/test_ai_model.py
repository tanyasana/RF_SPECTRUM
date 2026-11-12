"""
=============================================================================
Module: test_ai_model.py
Tests: PyTorch 1D-CNN Model Architecture & Differential Phase AMC Engine
=============================================================================
"""

import os
import sys
import numpy as np
import pytest
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from ai_model.model import ModulationCNN1D
from ai_model.dataset_builder import compute_differential_phase_features, generate_single_iq_frame
from ai_model.classifier import RFSignalClassifier


def test_model_forward_pass_shape():
    """Verify 1D-CNN input tensor (Batch, 4, 128) outputs logits of shape (Batch, 4)."""
    batch_size = 8
    seq_len = 128
    model = ModulationCNN1D(num_classes=4, input_channels=4, seq_len=seq_len)
    model.eval()

    dummy_input = torch.randn(batch_size, 4, seq_len, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)

    assert logits.shape == (batch_size, 4)
    assert probs.shape == (batch_size, 4)
    # Check probabilities sum to 1.0 per sample
    row_sums = torch.sum(probs, dim=-1).numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_differential_phase_invariance():
    """Verify that differential phase preprocessing preserves unit circle magnitude."""
    i_samples = np.random.randn(128).astype(np.float32)
    q_samples = np.random.randn(128).astype(np.float32)

    d_i, d_q = compute_differential_phase_features(i_samples, q_samples)
    assert len(d_i) == 128
    assert len(d_q) == 128
    # Boundary condition
    assert d_i[0] == 1.0
    assert d_q[0] == 0.0

    # cos^2 + sin^2 must equal ~1.0 for all n >= 1
    mags = np.sqrt(d_i[1:] ** 2 + d_q[1:] ** 2)
    assert np.allclose(mags, 1.0, atol=1e-4)


def test_classifier_trained_weights():
    """Verify RFSignalClassifier successfully loads weights and runs inference."""
    classifier = RFSignalClassifier()
    assert classifier.is_loaded is True

    # Test inference on a 1024-sample QPSK frame
    frame = generate_single_iq_frame(class_id=1, seq_len=1024, snr_db=20.0)
    i_samples = frame[0, :]
    q_samples = frame[1, :]

    result = classifier.predict_signal(i_samples, q_samples)
    assert "predicted_class" in result
    assert "confidence_pct" in result
    assert "shannon_entropy" in result
    assert result["confidence_pct"] >= 0.0
    assert result["predicted_class"] in ["BPSK", "QPSK", "8PSK", "16-QAM"]


def test_jammer_entropy_threat_gate():
    """Verify that pure Gaussian noise triggers the Shannon entropy threat gate."""
    # Pure noise (high ambiguity)
    noise_i = np.random.normal(0, 1.0, size=1024).astype(np.float32)
    noise_q = np.random.normal(0, 1.0, size=1024).astype(np.float32)

    classifier = RFSignalClassifier()
    res = classifier.predict_signal(noise_i, noise_q)

    # Entropy should be high for pure noise, tripping threat gate or anomaly flag
    assert res["shannon_entropy"] > 1.0 or res["is_anomaly"] is True
