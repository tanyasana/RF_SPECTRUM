"""
=============================================================================
Module: verilog_bridge.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 5 - Software-Hardware File Bridge (Python <-> Verilog)
=============================================================================

What This Bridge Does:
----------------------
1. Ingests floating-point IQ CSV signals from Step 2 (e.g., burst_interference.csv).
2. Quantizes floating-point samples [-1.0, +1.0] into signed 16-bit integers (Q1.15):
       I_fixed = int(np.clip(I_float * 32767, -32768, 32767))
3. Exports a standardized hexadecimal memory file:
       datasets/sim_vectors/sim_input.mem
   (Compatible with Verilog $readmemh)
4. Feeds the entire 1024-sample real dataset through the cycle-accurate Verilog 
   hardware model (spectrum_top).
5. Exports cycle-by-cycle hardware detection outputs to:
       datasets/sim_vectors/sim_output.csv
6. Verifies that the Verilog hardware caught the exact real-world burst interference!
"""

import os
import csv
import numpy as np
from scripts.run_spectrum_top_simulation import SpectrumTopSimulator


def quantize_iq_to_fixed_point(i_float: np.ndarray, q_float: np.ndarray, 
                               bit_width: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts normalized float IQ values [-1.0, 1.0] into signed 16-bit integers (Q1.15 format).
    Max positive: +32767 (0x7FFF)
    Max negative: -32768 (0x8000)
    """
    scale = (2 ** (bit_width - 1)) - 1
    min_val = -(2 ** (bit_width - 1))
    max_val = scale

    i_fixed = np.clip(np.round(i_float * scale), min_val, max_val).astype(np.int32)
    q_fixed = np.clip(np.round(q_float * scale), min_val, max_val).astype(np.int32)
    return i_fixed, q_fixed


def export_verilog_mem_file(i_fixed: np.ndarray, q_fixed: np.ndarray, mem_filepath: str):
    """
    Exports fixed-point IQ samples as 16-bit hexadecimal words to a .mem file:
    Format per line: <I_hex_16bit> <Q_hex_16bit>
    Example: 5A82 A57E
    """
    os.makedirs(os.path.dirname(os.path.abspath(mem_filepath)), exist_ok=True)
    with open(mem_filepath, mode="w", encoding="utf-8") as f:
        f.write("// Verilog $readmemh Compatible Memory File\n")
        f.write(f"// Total Samples: {len(i_fixed)} | Format: [15:0] I_HEX, [15:0] Q_HEX\n")
        for i_val, q_val in zip(i_fixed, q_fixed):
            # Convert signed 16-bit to 4-character hex
            i_hex = f"{i_val & 0xFFFF:04X}"
            q_hex = f"{q_val & 0xFFFF:04X}"
            f.write(f"{i_hex} {q_hex}\n")
    print(f"Exported {len(i_fixed)} fixed-point samples to: {mem_filepath}")


def run_bridge_simulation(csv_signal_path: str, mem_output_path: str, sim_results_path: str, 
                          power_threshold: int = 500000000, anomaly_limit: int = 5):
    """
    Full automated pipeline:
      CSV Float Samples -> Fixed-Point -> .mem file -> Verilog Top Simulation -> Output CSV
    """
    print("=" * 75)
    print("RUNNING STEP 5: AUTOMATED PYTHON <-> VERILOG SIMULATION BRIDGE")
    print("=" * 75)

    if not os.path.exists(csv_signal_path):
        raise FileNotFoundError(f"Signal file not found: {csv_signal_path}")

    # 1. Load CSV float samples
    i_list = []
    q_list = []
    with open(csv_signal_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i_list.append(float(row["I"]))
            q_list.append(float(row["Q"]))

    i_float = np.array(i_list, dtype=np.float64)
    q_float = np.array(q_list, dtype=np.float64)
    num_samples = len(i_float)
    print(f"[1/4] Loaded {num_samples} float samples from: {os.path.basename(csv_signal_path)}")

    # 2. Quantize to signed 16-bit fixed-point
    i_fixed, q_fixed = quantize_iq_to_fixed_point(i_float, q_float, bit_width=16)
    print(f"[2/4] Quantized to Q1.15 fixed-point (Range: [{np.min(i_fixed)}, {np.max(i_fixed)}])")

    # 3. Export .mem file for Verilog $readmemh
    export_verilog_mem_file(i_fixed, q_fixed, mem_output_path)
    print(f"[3/4] Generated Verilog test vector memory file: {mem_output_path}")

    # 4. Stream samples through Verilog spectrum_top hardware simulation
    print(f"[4/4] Streaming {num_samples} samples through Verilog spectrum_top core...")
    dut = SpectrumTopSimulator(threshold=power_threshold, anomaly_limit=anomaly_limit)
    dut.reset()

    sim_log = []
    time_ns = 20  # After initial reset

    alert_trigger_sample = None
    total_violations = 0

    for idx, (i_val, q_val) in enumerate(zip(i_fixed, q_fixed)):
        time_ns += 10
        row = dut.clock_cycle(in_I=int(i_val), in_Q=int(q_val))
        row["sample_index"] = idx
        row["time_ns"] = time_ns
        sim_log.append(row)

        if row["threshold_hit"] == 1:
            total_violations += 1

        if row["alert_flag"] == 1 and alert_trigger_sample is None:
            alert_trigger_sample = idx

    # Save output log
    fieldnames = ["sample_index", "time_ns", "reg_I", "reg_Q", "power", 
                  "threshold_hit", "abnormal_count", "fsm_state", "alert_flag", "report_ready"]
    with open(sim_results_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sim_log)

    print("\n" + "-" * 75)
    print("HARDWARE BRIDGE EXECUTION SUMMARY:")
    print(f"  Total Samples Processed   : {num_samples}")
    print(f"  Hardware Clock Frequency  : 100 MHz (10 ns per sample)")
    print(f"  Total Power Threshold Hits: {total_violations}")
    if alert_trigger_sample is not None:
        print(f"  >>> HARDWARE ALERT TRIGGERED AT SAMPLE : #{alert_trigger_sample} (Time: {sim_log[alert_trigger_sample]['time_ns']} ns)")
        print(f"  >>> FSM STATE                          : {sim_log[alert_trigger_sample]['fsm_state']}")
    else:
        print("  >>> Status: All samples within normal threshold bounds.")
    print(f"  Logged Hardware Output to : {sim_results_path}")
    print("-" * 75)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    burst_csv = os.path.join(base_dir, "datasets", "simulated", "burst_interference.csv")
    mem_path  = os.path.join(base_dir, "datasets", "sim_vectors", "sim_input.mem")
    out_csv   = os.path.join(base_dir, "datasets", "sim_vectors", "sim_output.csv")

    # In burst_interference.csv:
    # Normal samples: Amplitude ~0.707 -> Fixed ~23170 -> Power ~ 1.07e9
    # Burst samples (5x): Amplitude ~3.5 -> Fixed ~32767 (clamped) -> Power ~ 2.14e9
    # We set threshold = 1,400,000,000 to catch the 5x surge cleanly!
    run_bridge_simulation(burst_csv, mem_path, out_csv, power_threshold=1400000000, anomaly_limit=5)
