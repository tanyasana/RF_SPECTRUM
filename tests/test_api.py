"""
=============================================================================
Module: test_api.py
Tests: FastAPI REST Microservice Endpoints
=============================================================================
"""

import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from api.server import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 OK and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "supported_modulations" in data


def test_classify_signal_endpoint():
    """Verify /api/v1/classify/signal accepts raw IQ arrays and returns valid prediction."""
    # Generate 128 samples of simple I/Q
    i_samples = list(np.random.randn(128).astype(float))
    q_samples = list(np.random.randn(128).astype(float))

    payload = {
        "i_samples": i_samples,
        "q_samples": q_samples,
        "power_threshold": 2.5
    }
    response = client.post("/api/v1/classify/signal", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence_pct" in data
    assert "shannon_entropy" in data
    assert "is_anomaly" in data
    assert data["predicted_class"] in ["BPSK", "QPSK", "8PSK", "16-QAM"]


def test_classify_signal_validation_error():
    """Verify endpoint rejects mismatched array lengths."""
    payload = {
        "i_samples": [0.1, 0.2],
        "q_samples": [0.1]
    }
    response = client.post("/api/v1/classify/signal", json=payload)
    assert response.status_code == 400


def test_fft_analysis_endpoint():
    """Verify /api/v1/dsp/fft computes spectrum metrics."""
    i_samples = list(np.random.randn(512).astype(float))
    q_samples = list(np.random.randn(512).astype(float))

    payload = {
        "i_samples": i_samples,
        "q_samples": q_samples,
        "sample_rate": 1e6,
        "window": "hann"
    }
    response = client.post("/api/v1/dsp/fft", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "num_bins" in data
    assert "peak_frequency_khz" in data
    assert "occupancy_ratio_pct" in data


def test_verilog_simulate_endpoint():
    """Verify /api/v1/verilog/simulate runs RTL cycle-accurate emulation."""
    i_samples = [1.0] * 20
    q_samples = [1.0] * 20

    payload = {
        "i_samples": i_samples,
        "q_samples": q_samples,
        "power_threshold_fixed": 10000000,
        "anomaly_limit": 5
    }
    response = client.post("/api/v1/verilog/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_cycles"] == 20
    assert "threshold_violations" in data
    assert "final_fsm_state" in data
