"""
=============================================================================
Utility: rf_cli.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: CSE Software Architecture - Unified Multi-Command CLI Tool
Author: Akshay Kumar (CSE Collaborator)
=============================================================================

Usage Examples:
  python rf_cli.py verify                  # Run master 5-stage regression suite
  python rf_cli.py audit                   # Run 5-stage forensic ML audit
  python rf_cli.py classify <csv_path>     # Classify an IQ dataset file
  python rf_cli.py serve --port 8000       # Launch FastAPI REST microservice
  python rf_cli.py dashboard               # Launch Streamlit web HMI console
  python rf_cli.py export                  # Export TorchScript JIT & ONNX edge models
  python rf_cli.py simulate-rtl            # Run cycle-accurate Verilog SoC emulation
  python rf_cli.py test                    # Run automated pytest test suite
"""

import os
import sys
import argparse
import subprocess

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def cmd_verify(args):
    print(">> Launching Master 6-Stage System Verification Suite...")
    from scripts.run_full_system_verification import run_full_integration as run_verify
    run_verify()


def cmd_audit(args):
    print(">> Launching Forensic ML Audit Engine...")
    from scripts.audit_model_authenticity import run_audit
    run_audit()


def cmd_classify(args):
    csv_file = args.file
    if not os.path.exists(csv_file):
        print(f"Error: Target file not found: {csv_file}")
        sys.exit(1)
    from ai_model.classifier import classify_csv_file
    classify_csv_file(csv_file, power_threshold=args.threshold)


def cmd_serve(args):
    import uvicorn
    print(f">> Starting FastAPI REST microservice on http://{args.host}:{args.port}")
    print(f">> Interactive Swagger documentation available at: http://{args.host}:{args.port}/docs")
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)


def cmd_dashboard(args):
    print(f">> Launching Streamlit Web HMI Console on port {args.port}...")
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        os.path.join(BASE_DIR, "dashboard", "app.py"),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.headless else "false"
    ]
    subprocess.run(cmd)


def cmd_export(args):
    print(">> Launching CSE Edge Model Compiler (TorchScript JIT & ONNX)...")
    from ai_model.export_engine import export_models
    export_models()


def cmd_simulate_rtl(args):
    print(f">> Running Verilog Hardware SoC Emulation on: {args.csv}")
    from python.verilog_bridge import run_bridge_simulation
    mem_path = os.path.join(BASE_DIR, "datasets", "sim_vectors", "cli_sim_input.mem")
    out_csv = os.path.join(BASE_DIR, "datasets", "sim_vectors", "cli_sim_output.csv")
    run_bridge_simulation(args.csv, mem_path, out_csv)


def cmd_test(args):
    print(">> Executing Automated Pytest Test Suite...")
    import pytest
    tests_dir = os.path.join(BASE_DIR, "tests")
    exit_code = pytest.main(["-v", tests_dir])
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(
        prog="rf_cli",
        description="Unified Command-Line Interface for AI-Driven RF Spectrum Monitoring & Interference Detection System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # verify
    p_verify = subparsers.add_parser("verify", help="Run master 5-stage system verification suite")
    p_verify.set_defaults(func=cmd_verify)

    # audit
    p_audit = subparsers.add_parser("audit", help="Run 5-stage forensic ML data leakage audit")
    p_audit.set_defaults(func=cmd_audit)

    # classify
    p_classify = subparsers.add_parser("classify", help="Classify an IQ CSV capture file")
    p_classify.add_argument("file", help="Path to CSV signal file")
    p_classify.add_argument("--threshold", type=float, default=2.5, help="Anomalous power threshold (default: 2.5)")
    p_classify.set_defaults(func=cmd_classify)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI REST microservice")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    p_serve.set_defaults(func=cmd_serve)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit Web HMI console")
    p_dash.add_argument("--port", type=int, default=8501, help="Dashboard port (default: 8501)")
    p_dash.add_argument("--headless", action="store_true", help="Run in headless browser mode")
    p_dash.set_defaults(func=cmd_dashboard)

    # export
    p_export = subparsers.add_parser("export", help="Compile 1D-CNN to TorchScript JIT and ONNX")
    p_export.set_defaults(func=cmd_export)

    # simulate-rtl
    p_sim = subparsers.add_parser("simulate-rtl", help="Run Verilog SoC cycle-accurate RTL emulation")
    p_sim.add_argument(
        "--csv",
        default=os.path.join(BASE_DIR, "datasets", "simulated", "burst_interference.csv"),
        help="Input CSV file for hardware emulation"
    )
    p_sim.set_defaults(func=cmd_simulate_rtl)

    # test
    p_test = subparsers.add_parser("test", help="Run pytest automated test suite")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
