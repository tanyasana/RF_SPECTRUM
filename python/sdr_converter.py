"""
=============================================================================
Module: sdr_converter.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Utility: Over-the-Air SDR Binary Capture to System CSV Ingestion Converter
=============================================================================

This utility bridges physical Software-Defined Radio (SDR) hardware captures
with our Python DSP, Verilog SoC, and 1D-CNN AMC pipeline:

Supported SDR Hardware Formats:
  1. 'rtl_sdr'   : RTL-SDR 8-bit unsigned interleaved IQ (uint8: 0..255)
  2. 'hackrf'    : HackRF One / BladeRF 8-bit signed interleaved IQ (int8: -128..127)
  3. 'usrp_sc16' : Ettus USRP / LimeSDR 16-bit signed interleaved IQ (int16: -32768..32767)
  4. 'gnuradio'  : GNU Radio / MATLAB 32-bit complex float interleaved (complex64 / float32)

Usage:
  python python/sdr_converter.py <input.bin> <output.csv> --format rtl_sdr --max-samples 2048
"""

import os
import sys
import csv
import argparse
import numpy as np


def convert_sdr_binary_to_iq(binary_filepath: str, 
                             sdr_format: str = "rtl_sdr", 
                             max_samples: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads raw binary SDR capture files and converts them into normalized float32 I and Q arrays.
    """
    if not os.path.exists(binary_filepath):
        raise FileNotFoundError(f"Binary capture file not found: {binary_filepath}")

    format_lower = sdr_format.lower()

    if format_lower == "rtl_sdr":
        # 8-bit unsigned interleaved [I0, Q0, I1, Q1, ...]
        raw = np.fromfile(binary_filepath, dtype=np.uint8, count=max_samples * 2)
        if len(raw) < 2:
            raise ValueError("Capture file contains insufficient bytes.")
        i_raw = raw[0::2].astype(np.float32)
        q_raw = raw[1::2].astype(np.float32)
        # Center around 0 (-1.0 to +1.0)
        i_samples = (i_raw - 127.5) / 127.5
        q_samples = (q_raw - 127.5) / 127.5

    elif format_lower in ["hackrf", "bladerf", "int8"]:
        # 8-bit signed interleaved [I0, Q0, ...]
        raw = np.fromfile(binary_filepath, dtype=np.int8, count=max_samples * 2)
        i_samples = raw[0::2].astype(np.float32) / 128.0
        q_samples = raw[1::2].astype(np.float32) / 128.0

    elif format_lower in ["usrp_sc16", "usrp", "int16"]:
        # 16-bit signed interleaved [I0, Q0, ...]
        raw = np.fromfile(binary_filepath, dtype=np.int16, count=max_samples * 2)
        i_samples = raw[0::2].astype(np.float32) / 32768.0
        q_samples = raw[1::2].astype(np.float32) / 32768.0

    elif format_lower in ["gnuradio", "fc32", "complex64"]:
        # 32-bit float complex interleaved
        raw = np.fromfile(binary_filepath, dtype=np.complex64, count=max_samples)
        i_samples = np.real(raw).astype(np.float32)
        q_samples = np.imag(raw).astype(np.float32)

    else:
        raise ValueError(f"Unsupported SDR format '{sdr_format}'. Choose from: rtl_sdr, hackrf, usrp_sc16, gnuradio")

    # RMS Energy Normalization
    rms = np.sqrt(np.mean(i_samples**2 + q_samples**2))
    if rms > 0:
        i_samples = i_samples / rms
        q_samples = q_samples / rms

    return i_samples, q_samples


def export_iq_to_csv(i_samples: np.ndarray, q_samples: np.ndarray, output_csv: str, label: str = "OTA_SDR_CAPTURE"):
    """
    Exports normalized I and Q arrays to standard system CSV format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    num_samples = min(len(i_samples), len(q_samples))

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "I", "Q", "magnitude", "power", "label"])
        for idx in range(num_samples):
            i_val = float(i_samples[idx])
            q_val = float(q_samples[idx])
            mag = float(np.sqrt(i_val**2 + q_val**2))
            pwr = float(i_val**2 + q_val**2)
            writer.writerow([idx, f"{i_val:.6f}", f"{q_val:.6f}", f"{mag:.6f}", f"{pwr:.6f}", label])

    print(f"Exported {num_samples} SDR samples to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Convert over-the-air SDR binary capture files to system CSV format.")
    parser.add_argument("input_bin", type=str, help="Path to raw SDR binary capture file (.bin/.dat)")
    parser.add_argument("output_csv", type=str, help="Destination path for formatted CSV file (.csv)")
    parser.add_argument("--format", type=str, default="rtl_sdr", 
                        choices=["rtl_sdr", "hackrf", "usrp_sc16", "gnuradio"],
                        help="SDR hardware binary encoding format")
    parser.add_argument("--max-samples", type=int, default=1024, help="Maximum number of IQ samples to extract")
    parser.add_argument("--label", type=str, default="SDR_LIVE_CAPTURE", help="Dataset identifier tag")

    args = parser.parse_args()
    i_s, q_s = convert_sdr_binary_to_iq(args.input_bin, sdr_format=args.format, max_samples=args.max_samples)
    export_iq_to_csv(i_s, q_s, args.output_csv, label=args.label)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Self-test demonstration with synthetic RTL-SDR binary
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dummy_bin = os.path.join(test_dir, "datasets", "sim_vectors", "sample_rtlsdr_capture.bin")
        dummy_csv = os.path.join(test_dir, "datasets", "sim_vectors", "sample_rtlsdr_converted.csv")
        
        # Generate 1024 samples of uint8 RTL-SDR interleaved data
        t = np.linspace(0, 1, 1024)
        synth_i = (127.5 + 80.0 * np.cos(2 * np.pi * 5 * t)).astype(np.uint8)
        synth_q = (127.5 + 80.0 * np.sin(2 * np.pi * 5 * t)).astype(np.uint8)
        interleaved = np.empty((2048,), dtype=np.uint8)
        interleaved[0::2] = synth_i
        interleaved[1::2] = synth_q
        
        with open(dummy_bin, "wb") as f:
            f.write(interleaved.tobytes())
            
        print("[SELF-TEST] Created synthetic RTL-SDR binary capture.")
        i_arr, q_arr = convert_sdr_binary_to_iq(dummy_bin, sdr_format="rtl_sdr", max_samples=1024)
        export_iq_to_csv(i_arr, q_arr, dummy_csv, label="RTL_SDR_DEMO")
        print(f"[SELF-TEST SUCCESS] Verified RTL-SDR conversion. RMS power: {np.mean(i_arr**2 + q_arr**2):.4f}")
    else:
        main()
