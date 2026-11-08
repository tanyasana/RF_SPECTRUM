# 📡 AI-Driven RF Spectrum Monitoring & Interference Detection System
### A Dual-Domain Hardware-Software Co-Design Architecture (FPGA RTL + Deep Learning AMC)

[![System Status](https://img.shields.io/badge/System%20Status-100%25%20Verified%20%26%20Operational-brightgreen?style=for-the-badge)](#)
[![Verilog](https://img.shields.io/badge/Hardware-Synthesizable%20Verilog%20RTL-purple?style=for-the-badge)](#)
[![Python](https://img.shields.io/badge/DSP%20%26%20Software-Python%203.14-blue?style=for-the-badge)](#)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%201D--CNN-red?style=for-the-badge)](#)
[![Dashboard](https://img.shields.io/badge/Console-Streamlit%20Web%20HMI-orange?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📑 Table of Contents
1. [Executive Overview](#-executive-overview)
2. [👥 Two-Engineer Project Division & Responsibilities](#-two-engineer-project-division--responsibilities)
   - [⚡ ECE Student: Lead Hardware, RF Physical Layer & DSP Architect (Major Highlight)](#-ece-student-lead-hardware-rf-physical-layer--dsp-architect-major-highlight)
   - [💻 CSE Student: Machine Learning, Software Architecture & Cloud/HMI Engineer](#-cse-student-machine-learning-software-architecture--cloudhmi-engineer)
3. [System Architecture Diagram](#-system-architecture-diagram)
4. [Hardware Subsystem: Synthesizable Verilog RTL (ECE)](#-hardware-subsystem-synthesizable-verilog-rtl-ece)
5. [Digital Signal Processing (DSP) Subsystem (ECE)](#-digital-signal-processing-dsp-subsystem-ece)
6. [Machine Learning Subsystem: 1D-CNN AMC & Jammer Gate (CSE)](#-machine-learning-subsystem-1d-cnn-amc--jammer-gate-cse)
7. [Streamlit Interactive Web Console (CSE)](#-streamlit-interactive-web-console-cse)
8. [FPGA Resource Utilization & Performance Benchmarks](#-fpga-resource-utilization--performance-benchmarks)
9. [Zero Data-Leakage ML Audit & Authenticity Report](#-zero-data-leakage-ml-audit--authenticity-report)
10. [Repository File & Directory Structure](#-repository-file--directory-structure)
11. [Quick Start & Reproduction Guide](#-quick-start--reproduction-guide)
12. [Presentation & Capstone Defense Deliverables](#-presentation--capstone-defense-deliverables)

---

## 🚀 Executive Overview

In congested RF environments (military communications, 5G/6G cellular, aerospace, and satellite networks), wireless spectrum is heavily susceptible to intentional jamming, unauthorized access, and physical-layer interference. 

Conventional spectrum monitoring architectures suffer from a critical trade-off:
* **Pure Software Approaches (SDRs on General Purpose Processors):** Highly cognitive and flexible, but hindered by operating system jitter, thread context switching, and latencies in the **tens of milliseconds**—too slow to protect sensitive RF front-ends from microsecond burst attacks.
* **Pure Hardware Approaches (Traditional FPGAs/ASICs):** Sub-microsecond deterministic speed, but rigid and unable to recognize complex non-stationary modulation patterns or dynamically adapt to emerging threats.

### The Innovation: Dual-Domain Hardware-Software Co-Design
This project presents an end-to-end, dual-domain cognitive spectrum monitoring system that fuses:
1. **The Deterministic Fast Path (Synthesizable Verilog FPGA RTL):** A cycle-accurate digital circuit executing fixed-point instantaneous power arithmetic ($P = I^2 + Q^2$), continuous comparator checks, and a 4-state Moore Finite State Machine (FSM). It reacts to high-power energy surges in **under 5 clock cycles ($< 50\text{ ns}$)**.
2. **The Cognitive Smart Path (1D-CNN + Differential Phase Preprocessing):** A 4-channel temporal Convolutional Neural Network that classifies complex modulation schemes (BPSK, QPSK, 8PSK, 16-QAM) with **$99.2\%$ average high-SNR accuracy** in **$1.02\text{ ms}$**, backed by a Shannon entropy and kurtosis anomaly gate to catch adversarial broadband jammers.

---

## 👥 Two-Engineer Project Division & Responsibilities

This interdisciplinary capstone project was designed, implemented, and verified collaboratively between an **Electronics & Communication Engineering (ECE)** student and a **Computer Science & Engineering (CSE)** student.

```
+---------------------------------------------------------------------------------------+
|                       AI-DRIVEN RF SPECTRUM MONITORING SYSTEM                         |
+-------------------------------------------+-------------------------------------------+
|      ⚡ ECE STUDENT RESPONSIBILITIES      |      💻 CSE STUDENT RESPONSIBILITIES      |
|  Hardware, RF Physical Layer & DSP Lead   |   AI, Software Architecture & Web HMI     |
+-------------------------------------------+-------------------------------------------+
| • RF Baseband & IQ Physical Layer Physics | • 1D-CNN PyTorch Neural Network Design    |
| • BPSK, QPSK, 8PSK, 16-QAM Constellations | • Differential Phase Tensor Preprocessing |
| • AWGN Channel & Threat Injection Modeling| • Cosine Annealing Training Pipeline      |
| • 1024-Point FFT & Welch PSD Math Pipeline| • Shannon Entropy & OOD Jammer Gate       |
| • 2D STFT Waterfall Spectrogram Transforms| • Statistical Crest Factor & Kurtosis Gate|
| • 6 Synthesizable Verilog RTL Modules     | • Streamlit Interactive Web HMI Console   |
| • Q1.15 Fixed-Point Pipelined Multipliers | • Plotly Dynamic Telemetry & Visualizers  |
| • 4-State Moore Hardware FSM Controller   | • Master 5-Stage System Verification Suite|
| • Cycle-Accurate Verilog Testbenches & VCD| • Forensic ML Data Leakage Audit Engine   |
| • FPGA Synthesis & Resource Utilization   | • Python-Verilog Telemetry Serialization  |
+-------------------------------------------+-------------------------------------------+
```

---

### ⚡ ECE Student: Lead Hardware, RF Physical Layer & DSP Architect (Major Highlight)

The **ECE Student** spearheaded the entire **physical layer foundations, digital signal processing mathematical formulation, synthesizable digital hardware design (Verilog RTL), FPGA synthesis targeting, and hardware-software bit-exact quantization**.

#### 1. RF Physical Layer & Signal Formulation
* Formulated the mathematical foundation of complex baseband IQ signals:
  $$s(t) = I(t)\cos(2\pi f_c t) - Q(t)\sin(2\pi f_c t) \implies \tilde{s}(t) = I(t) + jQ(t)$$
* Modeled 4 standard digital modulation schemes from first principles:
  * **BPSK:** 1 bit/symbol, 2 constellation states ($0^\circ, 180^\circ$).
  * **QPSK:** 2 bits/symbol, 4 orthogonal states ($45^\circ, 135^\circ, 225^\circ, 315^\circ$).
  * **8PSK:** 3 bits/symbol, 8 circular states spaced at $45^\circ$ increments.
  * **16-QAM:** 4 bits/symbol, $4 \times 4$ rectangular grid combining amplitude and phase modulation.
* Designed the physical models for electronic warfare interference:
  * **Pulsed Burst Interference:** Modeled high-energy transient RF bursts with rapid rise/fall times.
  * **Swept Chirp Jammer:** Modeled continuous Linear Frequency Modulation (LFM) sweeps across bandwidth $B$ with chirp rate $\mu = B / T_{\text{chirp}}$.
* Built the Additive White Gaussian Noise (AWGN) channel simulator with calibrated Signal-to-Noise Ratio (SNR) scaling from $-6\text{ dB}$ to $+24\text{ dB}$.

#### 2. Digital Signal Processing (DSP) Mathematical Pipeline
* **1024-Point Fast Fourier Transform (FFT):** Implemented radix-2 discrete Fourier transforms converting discrete time-domain symbols into 1024 frequency bins, calibrated to Decibels relative to Full Scale ($\text{dBFS}$).
* **Spectral Windowing & Leakage Mitigation:** Formulated Hann, Hamming, and Blackman tapering windows to suppress sidelobes and eliminate discontinuity leakage at frame boundaries.
* **Welch's Power Spectral Density (PSD):** Implemented segment-averaged periodograms with $50\%$ overlap to reduce variance and reveal true spectral peaks beneath noisy floors.
* **99% Occupied Bandwidth (OBW) & Channel Occupancy %:** Developed the cumulative power integration algorithm that dynamically calculates the exact bandwidth occupied by active signals versus background noise.
* **2D Short-Time Fourier Transform (STFT) Spectrogram:** Formulated the time-frequency transform matrix $S(t, f)$ with sliding windows to capture temporal energy dynamics.

#### 3. Synthesizable Verilog Digital System Design (FPGA RTL)
Architected, coded, and verified **6 synthesizable IEEE 1364-2001 Verilog modules** in `verilog/`:
* **`iq_register.v` (Input Synchronizer):** Pipelined sequential register latching 16-bit signed $I$ and $Q$ symbols on `posedge clk` with synchronous active-low reset and clock enable to prevent metastability.
* **`magnitude_calculator.v` (Pipelined Power Core):** Computes instantaneous signal power:
  $$P[n] = I[n]^2 + Q[n]^2$$
  Implemented using two parallel $16 \times 16 \rightarrow 32$-bit hardware multipliers and a 33-bit unsigned adder. **Key Engineering Decision:** Avoided a 16-cycle iterative CORDIC square root engine by comparing power directly against squared thresholds, reducing hardware latency from 18 cycles to 1 cycle!
* **`threshold_detector.v` (High-Speed Digital Comparator):** Performs cycle-by-cycle magnitude comparison of the 33-bit power against a programmable threshold $P_{\text{thresh}}$, asserting `over_threshold` in single-cycle logic.
* **`abnormal_counter.v` (Fault Accumulator with Hysteresis):** A saturating counter that increments on threshold violations and decrements/clears when normal, providing noise immunity and preventing false alarms from random thermal Gaussian spikes.
* **`detection_fsm.v` (4-State Moore Finite State Controller):** Implements an emergency hardware threat controller with states:
  * `IDLE (2'b00)`: Reset and initialization.
  * `MONITOR (2'b01)`: Active scanning of incoming IQ data stream.
  * `ALERT (2'b10)`: Consecutive fault limit reached; immediately asserts hardware interrupt line.
  * `REPORT (2'b11)`: Latches diagnostic registers for host CPU ingestion.
* **`spectrum_top.v` (SoC Top Wrapper):** Interconnects all modules into a unified synthesizable System-on-Chip (SoC) IP core.

#### 4. Hardware Quantization & Fixed-Point Verification
* Designed the **Q1.15 fixed-point quantization schema**: 1 sign bit, 15 fractional bits ($[-1.0, +0.9999]$ mapped to $[-32768, +32767]$).
* Created the bit-exact **Hardware-Software Bridge** (`python/verilog_bridge.py`), exporting hexadecimal simulation vectors (`sim_input.mem`) for Hardware-in-the-Loop (HIL) testbenches.
* Formulated single-file Verilog synthesis packages (`testbench/eda_playground_design.sv` and `testbench/eda_playground_testbench.sv`) for cloud-based verification on EDA Playground and waveform inspection via EPWave.
* Executed FPGA synthesis and timing analysis targeting the **Xilinx Artix-7 (XC7A100T)** FPGA:
  * Total Slice LUTs: **118** ($< 0.2\%$ utilization)
  * Slice Registers (FFs): **95** ($< 0.1\%$ utilization)
  * DSP48E1 Multiplier Slices: **2** (only $0.8\%$)
  * Maximum Operating Frequency ($f_{\max}$): **$> 150\text{ MHz}$** (Nanosecond reaction time)

---

### 💻 CSE Student: Machine Learning, Software Architecture & Cloud/HMI Engineer

The **CSE Student** designed and implemented the **Deep Learning 1D-CNN Automatic Modulation Classifier, the statistical anomaly gating engine, the master 5-stage regression verification suite, the forensic ML audit engine, and the full-stack Streamlit web application**.

#### 1. Deep Learning 1D-CNN Architecture
* Implemented a temporal 1D Convolutional Neural Network in PyTorch tailored for high-speed RF symbol sequences:
  * **Layer 1:** Conv1D (4 channels $\rightarrow$ 64 filters, kernel size 7, stride 1) + BatchNorm1d + ReLU + MaxPool1d(2).
  * **Layer 2:** Conv1D (64 $\rightarrow$ 128 filters, kernel size 5, stride 1) + BatchNorm1d + ReLU + MaxPool1d(2).
  * **Layer 3:** Conv1D (128 $\rightarrow$ 256 filters, kernel size 3, stride 1) + BatchNorm1d + ReLU + AdaptiveAvgPool1d(1).
  * **Classifier Head:** Linear(256 $\rightarrow$ 128) + ReLU + Dropout(0.3) + Linear(128 $\rightarrow$ 4) + Softmax.
* Engineered the **4-Channel Differential Phase Preprocessor**:
  * Augmented the raw input $[I(t), Q(t)]$ with differential phase transitions $[\cos(\Delta\phi(t)), \sin(\Delta\phi(t))]$ where $\Delta\phi(t) = \arg(s(t) \cdot s^*(t-1))$.
  * This made the network mathematically invariant to static carrier phase offsets and oscillator drift, eliminating classification ambiguity between QPSK ($90^\circ$ jumps) and 8PSK ($45^\circ$ jumps).
* Trained the network with **Cosine Annealing Learning Rate Scheduling** and Cross-Entropy Loss, achieving **$99.2\%$ average high-SNR accuracy** and an edge-capable inference latency of **$1.02\text{ ms}$**.

#### 2. Statistical Jamming Gate & Anomaly Detection
* Solved the traditional AI "overconfidence blindspot" where neural networks assign high confidence to out-of-distribution (OOD) waveforms.
* Built a multi-tier anomaly detection gate combining:
  * **Shannon Prediction Entropy:** $H = -\sum_{i=1}^M p_i \log_2(p_i)$. If entropy exceeds $1.2$, the frame is flagged as an unclassifiable jammer.
  * **Kurtosis & Crest Factor Gate:** Computes the peak-to-average power ratio (PAPR) and signal envelope kurtosis to catch non-standard pulsed and chirp attacks.

#### 3. Forensic ML Audit & Regression Testing
* Developed the master 5-stage verification suite (`scripts/run_full_system_verification.py`), validating end-to-end dataset integrity, DSP computations, Verilog cycle execution, AI inference, and HMI console readiness.
* Built the **5-stage Forensic ML Audit** (`scripts/audit_model_authenticity.py`), proving zero data leakage through label scrambling permutation tests (accuracy collapsed to the exact theoretical random floor of $25.0\%$).

#### 4. Streamlit Interactive Web HMI Console
* Designed and built the multi-tab real-time monitoring console (`dashboard/app.py`):
  * **Reactive Top KPI Banner:** Displays frame size, AI classification, threat status, Verilog FSM state, and spectrum occupancy %.
  * **5 Interactive Visualizer Tabs:** 1024-pt FFT power spectrum, 2D STFT waterfall heatmap, IQ constellation scatter plot, AI softmax distribution with entropy diagnostics, and cycle-by-cycle Verilog telemetry.
  * **Live Interactive Signal Synthesis:** Real-time waveform generator allowing operators to alter modulations, adjust SNR in dB, and inject rogue jammer bursts live.

---

## 🏛️ System Architecture Diagram

```
                                  [ RAW RF SIGNAL FEED ]
                               Analog RF Carrier via Antenna
                                             │
                                             ▼
                             [ RF FRONT-END DOWNCONVERSION ]
                               In-Phase (I) & Quadrature (Q)
                               16-bit ADC @ 20 MSPS Sampling
                                             │
                 ┌───────────────────────────┴───────────────────────────┐
                 │                                                       │
                 ▼                                                       ▼
    [ ECE: HARDWARE FAST PATH ]                             [ CSE: SOFTWARE SMART PATH ]
      Synthesizable Verilog RTL                               1D-CNN + Differential Phase
  ┌───────────────────────────────┐                       ┌───────────────────────────────┐
  │ iq_register.v                 │                       │ python/fft_analysis.py        │
  │ • Latch 16-bit Q1.15 I & Q    │                       │ • 1024-pt FFT & Welch PSD     │
  ├───────────────────────────────┤                       │ • 99% Occupied Bandwidth %    │
  │ magnitude_calculator.v        │                       ├───────────────────────────────┤
  │ • P = I² + Q² (2 Multipliers) │                       │ python/spectrogram.py         │
  ├───────────────────────────────┤                       │ • 2D STFT Waterfall Heatmap   │
  │ threshold_detector.v          │                       ├───────────────────────────────┤
  │ • P > P_thresh Comparator     │                       │ ai_model/dataset_builder.py   │
  ├───────────────────────────────┤                       │ • 4-Channel Tensor Preprocess │
  │ abnormal_counter.v            │                       │   [I, Q, cos(Δφ), sin(Δφ)]    │
  ├───────────────────────────────┤                       ├───────────────────────────────┤
  │ detection_fsm.v               │                       │ ai_model/model.py (1D-CNN)    │
  │ • Moore FSM (ALERT State)     │                       │ • BPSK, QPSK, 8PSK, 16-QAM    │
  └───────────────────────────────┘                       │ • Latency: 1.02 ms (99.2% Acc)│
                 │                                        ├───────────────────────────────┤
                 │ Hardware Interrupt Line (< 50 ns)      │ ai_model/classifier.py        │
                 │                                        │ • Shannon Entropy Threat Gate │
                 │                                        └───────────────────────────────┘
                 │                                                       │
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                             [ UNIFIED OPERATOR DASHBOARD ]
                                Streamlit Web HMI Console
                     Real-Time Spectrum, Telemetry & Jammer Alert
```

---

## 🔬 Hardware Subsystem: Synthesizable Verilog RTL (ECE)

The complete synthesizable Verilog core resides in `verilog/` and is structured as follows:

```
verilog/
├── iq_register.v           # 16-bit Q1.15 sequential sample register with enable
├── magnitude_calculator.v  # Instantaneous power arithmetic unit (P = I² + Q²)
├── threshold_detector.v    # 33-bit digital magnitude comparator
├── abnormal_counter.v      # Saturating fault counter with hysteresis filter
├── detection_fsm.v         # 4-state Moore Finite State Machine
└── spectrum_top.v          # Top-level SoC integration wrapper
```

### Module Interface & Signal Description
| Module | Primary Inputs | Primary Outputs | Latency | Function |
| :--- | :--- | :--- | :--- | :--- |
| `iq_register` | `clk, rst_n, en, i_in[15:0], q_in[15:0]` | `i_reg[15:0], q_reg[15:0]` | 1 Cycle | Metastability elimination & input latching |
| `magnitude_calculator` | `clk, rst_n, i_data[15:0], q_data[15:0]` | `p_out[32:0], valid_out` | 1 Cycle | Calculates $P = I^2 + Q^2$ using DSP multipliers |
| `threshold_detector` | `p_in[32:0], threshold[32:0]` | `over_threshold` | Combinational | Single-cycle magnitude comparison |
| `abnormal_counter` | `clk, rst_n, over_threshold, limit[7:0]` | `count[7:0], anomaly_limit_reached` | 1 Cycle | Fault accumulator with saturation & clear |
| `detection_fsm` | `clk, rst_n, anomaly_limit_reached` | `fsm_state[1:0], alert_pulse` | 1 Cycle | 4-state Moore threat latch (`ALERT / REPORT`) |
| `spectrum_top` | `clk, rst_n, en, i_in, q_in, thresh, limit` | `p_out, over_thresh, count, fsm_state` | Pipelined | Integrated SoC core with diagnostic telemetry |

---

## 📊 FPGA Resource Utilization & Performance Benchmarks

Targeting the **Xilinx Artix-7 (XC7A100TCSG324-1)** FPGA:

| Hardware Resource | Used by `spectrum_top` | Available on Artix-7 | Utilization Percentage | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Slice LUTs** | 118 | 63,400 | **0.19%** | PASS |
| **Slice Registers (FF)** | 95 | 126,800 | **0.07%** | PASS |
| **DSP48E1 Slices** | 2 | 240 | **0.83%** | PASS |
| **Block RAM (BRAM)** | 0 | 135 | **0.00%** | PASS |
| **Global Clock Buffers (BUFG)** | 1 | 32 | **3.13%** | PASS |
| **Maximum Frequency ($f_{\max}$)** | **184.2 MHz** | — | — | Meets Timing ($> 100\text{ MHz}$) |
| **Hardware Detection Latency** | **4 Clock Cycles** | — | **$21.7\text{ ns}$ @ 184 MHz** | Instant Protection |

---

## 🧠 Machine Learning Subsystem: 1D-CNN AMC & Jammer Gate (CSE)

### Neural Network Input Tensor Shape
$$\mathbf{X} \in \mathbb{R}^{B \times 4 \times 128}$$
* **Channel 0 ($I$):** Normalized In-Phase signal.
* **Channel 1 ($Q$):** Normalized Quadrature signal.
* **Channel 2 ($\Delta I$):** $\cos(\Delta\phi[n])$ — Phase transition velocity.
* **Channel 3 ($\Delta Q$):** $\sin(\Delta\phi[n])$ — Phase transition direction.

### Accuracy vs. SNR Benchmark Results
| SNR (dB) | BPSK Accuracy | QPSK Accuracy | 8PSK Accuracy | 16-QAM Accuracy | **Overall Average** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **-6 dB** | 82.4% | 61.2% | 48.0% | 51.5% | **60.8%** |
| **0 dB** | 98.1% | 84.7% | 76.3% | 79.2% | **84.6%** |
| **+6 dB** | 99.8% | 96.5% | 93.1% | 94.8% | **96.1%** |
| **+12 dB** | 100.0% | 99.4% | 98.7% | 98.9% | **99.2%** |
| **+18 dB** | 100.0% | 99.8% | 99.5% | 99.7% | **99.7%** |
| **+24 dB** | 100.0% | 100.0% | 99.7% | 99.8% | **99.9%** |

---

## 🛡️ Zero Data-Leakage ML Audit & Authenticity Report

To guarantee that the machine learning model did not overfit or memorize synthetic data artifacts, we executed a forensic audit (`scripts/audit_model_authenticity.py`):

1. **Permutation Test (Scrambled Labels):** Training labels were randomly shuffled. Validation accuracy immediately collapsed to **$25.1\%$** (the exact mathematical random guess probability $1/M = 25.0\%$), proving the model learns true physical RF geometry.
2. **Independent Out-of-Sample Test:** Evaluated on 4,000 completely fresh signals with random phase offsets, achieving **$98.83\%$** generalization accuracy.
3. **Inference Latency Audit:** Benchmarked over 1,000 continuous frames:
   * Mean Latency: **$1.02\text{ ms}$** per frame.
   * Peak Throughput: **$\approx 980\text{ frames/second}$**, confirming real-time edge streaming capability.

---

## 📁 Repository File & Directory Structure

```
RF_Spectrum_Monitoring_System/
├── verilog/                # [ECE] Synthesizable Verilog RTL Source Code
│   ├── iq_register.v       # 16-bit Q1.15 sequential capture register
│   ├── magnitude_calculator.v # Pipelined power arithmetic (P = I² + Q²)
│   ├── threshold_detector.v# High-speed digital magnitude comparator
│   ├── abnormal_counter.v  # Saturating fault counter with hysteresis
│   ├── detection_fsm.v     # 4-state Moore Finite State Controller
│   └── spectrum_top.v      # Integrated SoC top IP wrapper
│
├── python/                 # [ECE] Digital Signal Processing & Math Pipeline
│   ├── signal_generator.py # Synthesizes complex baseband IQ waveforms
│   ├── fft_analysis.py     # 1024-point FFT, Welch PSD & Occupancy %
│   ├── spectrogram.py      # 2D STFT time-frequency waterfall transforms
│   ├── anomaly_detection.py# Statistical kurtosis & crest factor estimators
│   └── verilog_bridge.py   # Fixed-point Q1.15 quantizer exporting sim_input.mem
│
├── testbench/              # [ECE] Verilog Testbenches & Simulation Packages
│   ├── tb_modules_1_to_4.v # Unit testbenches for arithmetic & comparator blocks
│   ├── tb_spectrum_top.v   # Top-level SoC testbench generating .vcd waveforms
│   ├── eda_playground_design.sv    # Cloud synthesis package (Design)
│   └── eda_playground_testbench.sv # Cloud testbench package (Testbench)
│
├── ai_model/               # [CSE] Deep Learning Modulation Classifier
│   ├── model.py            # 4-channel 1D-CNN PyTorch architecture
│   ├── dataset_builder.py  # Differential phase feature extraction tensor builder
│   ├── train.py            # Training pipeline with Cosine Annealing
│   ├── evaluate_benchmarks.py # Discrete SNR evaluation sweep (-6 dB to +24 dB)
│   ├── classifier.py       # Real-time inference engine with Shannon entropy gate
│   └── weights/            # Trained weights (best_rf_classifier.pt)
│
├── dashboard/              # [CSE] Streamlit Interactive Web HMI Console
│   └── app.py              # Full-stack reactive dashboard application
│
├── datasets/               # Signal Datasets & Hex Simulation Memory Vectors
│   ├── simulated/          # Generated IQ CSV datasets (Clean & Jammed)
│   └── sim_vectors/        # sim_input.mem (hex) & sim_output.csv (Verilog Bridge)
│
├── results/                # Evaluation Plots, Waveforms & Audit Reports
│   ├── plots/              # Spectral, waterfall, constellation & accuracy plots
│   └── reports/            # Master system verification reports
│
├── scripts/                # Master Automation & Verification Suites
│   ├── run_full_system_verification.py # Master 5-stage end-to-end regression suite
│   ├── audit_model_authenticity.py     # 5-stage forensic ML audit (zero leakage)
│   └── run_spectrum_top_simulation.py  # Cycle-by-cycle Verilog SoC hardware emulator
│
├── docs/                   # Documentation & Presentation Slides
│   └── presentation/       # 20-slide capstone defense presentation (.pptx)
│
├── .gitignore              # Production-grade git exclusion rules
├── requirements.txt        # Python dependency manifest
├── launch_dashboard.bat    # One-click Windows dashboard launcher
├── run_full_system_verification.bat # One-click Windows test suite launcher
└── README.md               # Master engineering documentation
```

---

## ⚡ Quick Start & Reproduction Guide

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/tanyasana/RF_SPECTRUM.git
cd RF_SPECTRUM
pip install -r requirements.txt
```

### 2. Run the Master 5-Stage Verification Suite
Execute the automated regression test validating all subsystems in under 1 second:
```bash
python scripts/run_full_system_verification.py
```
*(Windows users can simply double-click `run_full_system_verification.bat`)*

### 3. Launch the Interactive Web Dashboard
Start the real-time Streamlit HMI console:
```bash
python -m streamlit run dashboard/app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

### 4. Verify Verilog Hardware Simulation
* **Local Python Emulation:**
  ```bash
  python scripts/run_spectrum_top_simulation.py
  ```
* **Cloud EDA Playground Verification:**
  1. Open [EDA Playground](https://www.edaplayground.com/).
  2. Paste `testbench/eda_playground_design.sv` into the **left pane**.
  3. Paste `testbench/eda_playground_testbench.sv` into the **right pane**.
  4. Select **Icarus Verilog 0.10.0** and tick **Open EPWave after run**.
  5. Click **Run** to view live digital clock cycle waveforms.

---

## 🎓 Presentation & Capstone Defense Deliverables

A 20-slide widescreen presentation is included in `docs/presentation/`:
* **File:** `AI_Driven_RF_Spectrum_Monitoring_System.pptx`
* **Contents:** System motivation, mathematical derivations, Verilog circuit schematics, 1D-CNN neural network structure, differential phase geometry, confusion matrices, and live dashboard demonstration.

### Key Viva Defense Questions & Answers

#### For the ECE Student:
1. **Q: Why calculate $P = I^2 + Q^2$ instead of magnitude $\sqrt{I^2+Q^2}$ in hardware?**
   * *A: Calculating square root in synthesizable RTL requires either a 16-cycle iterative CORDIC engine or large look-up tables. By squaring the detection threshold ($P_{\text{thresh}} = \text{Threshold}^2$), we achieve single-cycle comparison using only two DSP multiplier slices, reducing latency from 18 cycles to 1 cycle.*
2. **Q: How does the Verilog FSM prevent false alarms from random thermal noise?**
   * *A: The `abnormal_counter.v` module implements a saturating accumulator with hysteresis. A single isolated noise spike does not trigger an alarm. The signal power must violate the threshold for $N$ consecutive clock cycles (e.g., 5 cycles) before the FSM transitions from `MONITOR` to `ALERT`.*

#### For the CSE Student:
1. **Q: Why use differential phase preprocessing instead of raw IQ samples?**
   * *A: Raw IQ constellations rotate in the complex plane due to Carrier Frequency Offset (CFO) and Doppler shift, causing neural networks to misclassify QPSK and 8PSK. Differential phase $\Delta\phi = \arg(s[n] \cdot s^*[n-1])$ isolates relative phase transitions, making the CNN invariant to static and slowly drifting carrier offsets.*
2. **Q: How does the system detect unknown jammers without hallucinating a valid modulation class?**
   * *A: We implemented a dual-stage out-of-distribution (OOD) gate using Shannon prediction entropy ($H = -\sum p_i \log_2 p_i$) and crest factor gating. When an unfamiliar jammer or noise burst is ingested, the prediction entropy spikes above $1.2$, triggering the threat gate and overriding the softmax classifier.*

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
