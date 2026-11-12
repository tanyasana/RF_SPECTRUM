"""
=============================================================================
Module: test_verilog_bridge.py
Tests: Q1.15 Fixed-Point Quantization & Verilog SoC RTL Emulation
=============================================================================
"""

import os
import sys
import numpy as np
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from python.verilog_bridge import quantize_iq_to_fixed_point
from scripts.run_spectrum_top_simulation import SpectrumTopSimulator


def test_q1_15_quantization_bounds():
    """Verify that floating-point inputs in [-1.0, 1.0] are mapped within [-32768, 32767]."""
    i_float = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    q_float = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    i_fixed, q_fixed = quantize_iq_to_fixed_point(i_float, q_float, bit_width=16)

    assert np.all(i_fixed >= -32768)
    assert np.all(i_fixed <= 32767)
    assert i_fixed[1] == -32767 or i_fixed[1] == -32768
    assert i_fixed[3] == 0
    assert i_fixed[5] == 32767


def test_verilog_simulator_reset_and_normal_operation():
    """Verify Verilog Moore FSM stays in MONITOR under nominal signal power."""
    sim = SpectrumTopSimulator(threshold=10000000, anomaly_limit=5)
    sim.reset()
    assert "MONITOR" in sim.fsm_state or "IDLE" in sim.fsm_state

    # Inject 5 small samples (Power = 1000^2 + 1000^2 = 2,000,000 < 10,000,000)
    for _ in range(5):
        res = sim.clock_cycle(1000, 1000)
        assert res["threshold_hit"] == 0
        assert res["alert_flag"] == 0


def test_verilog_simulator_burst_alert_transition():
    """Verify that continuous threshold violations drive the FSM into ALERT state."""
    sim = SpectrumTopSimulator(threshold=10000000, anomaly_limit=5)
    sim.reset()

    # Step through pipeline latency first (2 cycles)
    sim.clock_cycle(1000, 1000)
    sim.clock_cycle(1000, 1000)

    # Inject high power samples (Power = 4000^2 + 4000^2 = 32,000,000 > 10,000,000)
    alert_triggered = False
    for _ in range(15):
        res = sim.clock_cycle(4000, 4000)
        if res["alert_flag"] == 1 or "ALERT" in res["fsm_state"]:
            alert_triggered = True

    assert alert_triggered is True
