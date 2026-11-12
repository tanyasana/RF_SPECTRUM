"""
=============================================================================
Module: server.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: CSE Software Architecture - High-Throughput REST API Microservice
Author: Akshay Kumar (CSE Collaborator)
=============================================================================

This FastAPI service exposes the complete dual-domain RF monitoring pipeline
over a production-grade asynchronous REST API, allowing IoT edge sensors,
drone telemetry relays, and SDR receiver stations to stream IQ baseband data
and receive microsecond-level AI classifications and threat warnings.

Endpoints:
  - GET  /health               : System status, model loading, device info.
  - POST /api/v1/classify/signal : Classify raw I/Q array with Shannon entropy gate.
  - POST /api/v1/classify/file   : Upload CSV capture for full diagnostic report.
  - POST /api/v1/dsp/fft         : Compute 1024-pt FFT & spectrum occupancy ratio.
  - POST /api/v1/verilog/simulate: Run cycle-accurate Verilog FPGA hardware emulation.
"""

import os
import sys
import io
import time
import csv
from typing import List, Dict, Optional
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_model.classifier import RFSignalClassifier
from python.fft_analysis import compute_fft, compute_spectrum_occupancy
from python.anomaly_detection import compute_statistical_rf_metrics
from scripts.run_spectrum_top_simulation import SpectrumTopSimulator
from python.verilog_bridge import quantize_iq_to_fixed_point


# Initialize FastAPI application
app = FastAPI(
    title="AI-Driven RF Spectrum Monitoring Microservice",
    description="Asynchronous edge/cloud REST API for real-time automatic modulation classification (AMC) and physical-layer RF threat gating.",
    version="2.0.0",
    contact={
        "name": "Akshay Kumar (CSE Collaborator)",
        "email": "akshaykumarhudedmani@gmail.com"
    }
)

# Enable CORS for web dashboards and distributed sensor clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
_classifier: Optional[RFSignalClassifier] = None


def get_classifier() -> RFSignalClassifier:
    global _classifier
    if _classifier is None:
        _classifier = RFSignalClassifier()
    return _classifier


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class SignalInput(BaseModel):
    i_samples: List[float] = Field(..., description="In-Phase (I) baseband samples array")
    q_samples: List[float] = Field(..., description="Quadrature (Q) baseband samples array")
    power_threshold: float = Field(2.5, description="Instantaneous power threshold for energy burst anomaly")
    sample_rate: float = Field(1e6, description="Sampling rate in Hz (default: 1.0 MHz)")


class PredictionResponse(BaseModel):
    predicted_class: str
    class_id: int
    confidence_pct: float
    probabilities: Dict[str, float]
    shannon_entropy: float
    is_anomaly: bool
    anomaly_reason: str
    avg_power: float
    max_power: float
    papr_db: float
    kurtosis: float
    inference_latency_ms: float
    frames_analyzed: int


class FFTInput(BaseModel):
    i_samples: List[float]
    q_samples: List[float]
    sample_rate: float = 1e6
    window: str = "hann"
    occupancy_cutoff_db: float = -20.0


class FFTResponse(BaseModel):
    num_bins: int
    peak_frequency_khz: float
    occupied_bandwidth_khz: float
    occupancy_ratio_pct: float
    noise_floor_dbfs: float


class VerilogSimInput(BaseModel):
    i_samples: List[float]
    q_samples: List[float]
    power_threshold_fixed: int = Field(671088640, description="Q1.15 squared power threshold (default: 2.5 normalized)")
    anomaly_limit: int = Field(5, description="Consecutive fault accumulator trigger limit")


class VerilogSimResponse(BaseModel):
    total_cycles: int
    threshold_violations: int
    final_fsm_state: str
    alert_triggered: bool


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """Returns microservice health, model readiness, and hardware acceleration status."""
    clf = get_classifier()
    import torch
    return {
        "status": "healthy",
        "service": "RF_SPECTRUM_MONITORING_ENGINE",
        "version": "2.0.0",
        "model_loaded": clf.is_loaded,
        "device": str(clf.device),
        "pytorch_version": torch.__version__,
        "supported_modulations": ["BPSK", "QPSK", "8PSK", "16-QAM"]
    }


@app.post("/api/v1/classify/signal", response_model=PredictionResponse, tags=["Inference"])
def classify_signal(payload: SignalInput):
    """
    Ingests raw baseband I/Q sample arrays and returns real-time AMC predictions,
    Shannon entropy ambiguity metrics, and RF threat gating verdicts.
    """
    if len(payload.i_samples) != len(payload.q_samples):
        raise HTTPException(status_code=400, detail="In-Phase and Quadrature arrays must have identical length.")
    if len(payload.i_samples) < 8:
        raise HTTPException(status_code=400, detail="Minimum of 8 samples required for analysis.")

    i_arr = np.array(payload.i_samples, dtype=np.float32)
    q_arr = np.array(payload.q_samples, dtype=np.float32)

    clf = get_classifier()
    t0 = time.perf_counter()
    res = clf.predict_signal(i_arr, q_arr, power_threshold=payload.power_threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    stat = compute_statistical_rf_metrics(i_arr, q_arr)

    return {
        "predicted_class": res["predicted_class"],
        "class_id": res["class_id"],
        "confidence_pct": res["confidence_pct"],
        "probabilities": res["probabilities"],
        "shannon_entropy": res["shannon_entropy"],
        "is_anomaly": res["is_anomaly"],
        "anomaly_reason": res["anomaly_reason"],
        "avg_power": res["avg_power"],
        "max_power": res["max_power"],
        "papr_db": stat["papr_db"],
        "kurtosis": stat["kurtosis"],
        "inference_latency_ms": round(latency_ms, 3),
        "frames_analyzed": res["frames_analyzed"]
    }


@app.post("/api/v1/classify/file", response_model=PredictionResponse, tags=["Inference"])
async def classify_file(
    file: UploadFile = File(...),
    power_threshold: float = Query(2.5, description="Power threshold for burst anomaly")
):
    """
    Upload an IQ CSV file (columns: sample_index, I, Q) to receive full classification diagnostics.
    """
    content = await file.read()
    try:
        decoded = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))
        i_list = []
        q_list = []
        for row in reader:
            i_list.append(float(row["I"]))
            q_list.append(float(row["Q"]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")

    if len(i_list) == 0:
        raise HTTPException(status_code=400, detail="CSV file contains no valid sample data.")

    return classify_signal(SignalInput(
        i_samples=i_list,
        q_samples=q_list,
        power_threshold=power_threshold
    ))


@app.post("/api/v1/dsp/fft", response_model=FFTResponse, tags=["Digital Signal Processing"])
def analyze_fft(payload: FFTInput):
    """
    Executes a 1024-point Fast Fourier Transform (FFT) and computes spectral occupancy ratio.
    """
    if len(payload.i_samples) != len(payload.q_samples):
        raise HTTPException(status_code=400, detail="I and Q arrays must have matching dimensions.")

    complex_sig = np.array(payload.i_samples, dtype=np.float64) + 1j * np.array(payload.q_samples, dtype=np.float64)
    freqs, psd_db, _ = compute_fft(complex_sig, sample_rate=payload.sample_rate, window=payload.window)
    occ_pct, occ_mask, _ = compute_spectrum_occupancy(psd_db, threshold_db=payload.occupancy_cutoff_db)

    peak_idx = int(np.argmax(psd_db))
    peak_khz = float(freqs[peak_idx] / 1e3)
    bw_hz = len(psd_db) * (payload.sample_rate / len(psd_db))
    occ_bw_khz = float((bw_hz * occ_pct / 100.0) / 1e3)

    return {
        "num_bins": len(psd_db),
        "peak_frequency_khz": round(peak_khz, 2),
        "occupied_bandwidth_khz": round(occ_bw_khz, 2),
        "occupancy_ratio_pct": round(occ_pct, 2),
        "noise_floor_dbfs": round(float(np.percentile(psd_db, 10)), 2)
    }


@app.post("/api/v1/verilog/simulate", response_model=VerilogSimResponse, tags=["Hardware RTL Emulation"])
def simulate_verilog_hardware(payload: VerilogSimInput):
    """
    Emulates the 6-module synthesizable Verilog RTL SoC cycle-by-cycle in software.
    """
    if len(payload.i_samples) != len(payload.q_samples):
        raise HTTPException(status_code=400, detail="I and Q arrays must have matching dimensions.")

    i_arr = np.array(payload.i_samples, dtype=np.float32)
    q_arr = np.array(payload.q_samples, dtype=np.float32)

    i_fixed, q_fixed = quantize_iq_to_fixed_point(i_arr, q_arr)

    sim = SpectrumTopSimulator(
        threshold=payload.power_threshold_fixed,
        anomaly_limit=payload.anomaly_limit
    )

    violations = 0
    final_state = "IDLE"
    alert = False

    for i_val, q_val in zip(i_fixed, q_fixed):
        step_res = sim.clock_cycle(int(i_val), int(q_val))
        if step_res["threshold_hit"]:
            violations += 1
        final_state = step_res["fsm_state"]
        if step_res["alert_flag"] or "ALERT" in step_res["fsm_state"] or "REPORT" in step_res["fsm_state"]:
            alert = True

    return {
        "total_cycles": len(i_fixed),
        "threshold_violations": violations,
        "final_fsm_state": final_state,
        "alert_triggered": alert
    }
