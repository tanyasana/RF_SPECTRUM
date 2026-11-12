# 📡 AI-Driven RF Spectrum Monitoring & Interference Detection System
### A Dual-Domain Hardware-Software Co-Design Architecture (Synthesizable FPGA RTL + Deep Learning AMC + Cloud REST API)

[![System Status](https://img.shields.io/badge/System%20Status-100%25%20Verified%20(6%20Stages)-brightgreen?style=for-the-badge)](#)
[![Tests](https://img.shields.io/badge/Pytest-17%2F17%20Passed%20(100%25)-success?style=for-the-badge)](#)
[![Verilog](https://img.shields.io/badge/Hardware-Synthesizable%20Verilog%20RTL-purple?style=for-the-badge)](#)
[![FastAPI](https://img.shields.io/badge/REST%20API-FastAPI%20%2B%20Swagger-009688?style=for-the-badge)](#)
[![TorchScript](https://img.shields.io/badge/Edge%20AI-TorchScript%20JIT%20(0.72%20ms)-red?style=for-the-badge)](#)
[![Dashboard](https://img.shields.io/badge/Console-Streamlit%20Web%20HMI-orange?style=for-the-badge)](#)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20%26%20Compose-2496ED?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📑 Table of Contents
1. [Executive Overview & Key Highlights](#-executive-overview--key-highlights)
2. [👥 Engineering Responsibilities & Architectural Roles](#-engineering-responsibilities--architectural-roles)
   - [⚡ Taniya Sana: Electronics & Communication Engineering (ECE) Lead](#-taniya-sana-electronics--communication-engineering-ece-lead)
   - [💻 Akshay Kumar Hudedmani: Computer Science & Engineering (CSE) Lead](#-akshay-kumar-hudedmani-computer-science--engineering-cse-lead)
3. [🏛️ Dual-Domain System Architecture](#-dual-domain-system-architecture)
4. [🖥️ Streamlit Interactive Web HMI Console (Live Screenshots)](#-streamlit-interactive-web-hmi-console-live-screenshots)
5. [🧠 Machine Learning Subsystem: 1D-CNN AMC & Edge Optimization](#-machine-learning-subsystem-1d-cnn-amc--edge-optimization)
   - [Model Architecture & Differential Phase Invariance](#model-architecture--differential-phase-invariance)
   - [Training Curves, Confusion Matrix & SNR Benchmarks](#training-curves-confusion-matrix--snr-benchmarks)
   - [TorchScript JIT Edge Compilation & Latency Profile](#torchscript-jit-edge-compilation--latency-profile)
   - [Shannon Entropy & Statistical Jamming Threat Gate](#shannon-entropy--statistical-jamming-threat-gate)
6. [🌐 FastAPI Asynchronous REST Microservice](#-fastapi-asynchronous-rest-microservice)
7. [🛠️ Unified Multi-Command CLI Utility (`rf_cli.py`)](#-unified-multi-command-cli-utility-rf_clipy)
8. [🔬 Hardware Subsystem: Synthesizable Verilog RTL (ECE)](#-hardware-subsystem-synthesizable-verilog-rtl-ece)
   - [RTL Core Modules & Signal Routing](#rtl-core-modules--signal-routing)
   - [FPGA Resource Utilization (Xilinx Artix-7)](#fpga-resource-utilization-xilinx-artix-7)
9. [📈 Digital Signal Processing (DSP) Pipeline (ECE)](#-digital-signal-processing-dsp-pipeline-ece)
   - [1024-Point FFT & Welch PSD Power Spectrum](#1024-point-fft--welch-psd-power-spectrum)
   - [2D STFT Waterfall Spectrograms](#2d-stft-waterfall-spectrograms)
   - [IQ Constellations & Differential Phase Space](#iq-constellations--differential-phase-space)
10. [🧪 Automated Verification & Pytest Suite (17/17 Passed)](#-automated-verification--pytest-suite-1717-passed)
11. [🛡️ Zero Data-Leakage Forensic ML Audit Report](#-zero-data-leakage-forensic-ml-audit-report)
12. [🐳 Docker Containerization & Deployment](#-docker-containerization--deployment)
13. [⚡ Quick Start & Reproduction Guide](#-quick-start--reproduction-guide)
14. [📁 Complete Repository Structure](#-complete-repository-structure)
15. [📜 License](#-license)

---

## 🚀 Executive Overview & Key Highlights

In congested electromagnetic environments (military electronic warfare, 5G/6G private networks, commercial aviation, and satellite uplinks), radio frequency spectrum is vulnerable to intentional jamming, malicious spoofing, and physical-layer interference. 

Conventional spectrum monitoring approaches face a fundamental engineering compromise:
* **Pure Software Radios (SDR on GPPs):** Agile and cognitive, but plagued by OS scheduling jitter, USB/PCIe buffering overhead, and latencies in the **10–50 ms** range—far too sluggish to protect RF low-noise amplifiers (LNAs) from microsecond burst energy attacks.
* **Pure Hardware ASICs/FPGAs:** Nanosecond deterministic reaction times, but structurally rigid and unable to recognize complex modulation schemes or dynamically adapt to non-stationary threats.

### The Innovation: Dual-Domain Hardware-Software Co-Design
This project presents an integrated, dual-domain cognitive RF monitoring architecture:
1. **The Deterministic Fast Path (Synthesizable Verilog FPGA RTL):** A cycle-accurate digital circuit executing fixed-point instantaneous power arithmetic ($P = I^2 + Q^2$), continuous comparator checks, and a 4-state Moore Finite State Machine (FSM). It reacts to high-power energy surges in **under 4 clock cycles ($< 50\text{ ns}$)**.
2. **The Cognitive Smart Path (1D-CNN + Differential Phase Preprocessing):** A 4-channel temporal Convolutional Neural Network classifying modulation schemes (BPSK, QPSK, 8PSK, 16-QAM) with **$99.2\%$ average high-SNR accuracy** and edge-optimized **$0.72\text{ ms}$ inference latency ($1380+\text{ FPS}$ via TorchScript JIT)**, backed by a Shannon entropy and kurtosis anomaly gate to catch adversarial out-of-distribution jammers.

### 📊 System Scorecard at a Glance
| Key Metric | Hardware Fast Path (ECE) | Cognitive Smart Path (CSE) |
| :--- | :--- | :--- |
| **Execution Domain** | Synthesizable IEEE 1364-2001 Verilog RTL | PyTorch 1D-CNN + TorchScript JIT + FastAPI |
| **Reaction Latency** | **$21.7\text{ ns}$** (4 clock cycles @ $184.2\text{ MHz}$) | **$0.72\text{ ms}$** per 128-sample frame ($1380+\text{ FPS}$) |
| **FPGA Footprint** | **118 Slice LUTs ($0.19\%$)**, 2 DSP48E1 on Artix-7 | Edge Deployable on CPU, Jetson, or RPi 5 |
| **Accuracy / Target** | $100\%$ detection of high-power energy surges | **$99.2\%$ average high-SNR accuracy** across 4 classes |
| **Threat Coverage** | Transient burst pulses, amplifier overdrive | LFM Chirp jammers, carrier offsets, thermal noise |
| **Interfaces** | Hardware Interrupt Line (`ALERT`), Q1.15 Bridge | REST API (Swagger), Streamlit Web Console, CLI |

---

## 👥 Engineering Responsibilities & Architectural Roles

This project is architected with a strict modular co-design division across **Electronics & Communication Engineering (ECE)** and **Computer Science & Engineering (CSE)**:

```
+---------------------------------------------------------------------------------------------------------+
|                                AI-DRIVEN RF SPECTRUM MONITORING SYSTEM                                  |
+----------------------------------------------------+----------------------------------------------------+
|       ⚡ TANIYA SANA (ECE LEAD & ARCHITECT)        |     💻 AKSHAY KUMAR HUDEDMANI (CSE COLLABORATOR)   |
|       Hardware RTL, RF Physical Layer & DSP        |      AI, Software Architecture, REST API & HMI     |
+----------------------------------------------------+----------------------------------------------------+
| • RF Complex Baseband IQ Physics Formulation       | • 1D-CNN PyTorch Neural Network Architecture       |
| • BPSK, QPSK, 8PSK, 16-QAM Constellation Modeling  | • Differential Phase Invariant Feature Extraction  |
| • AWGN Channel Simulation (-6 dB to +24 dB SNR)    | • Cosine Annealing Learning Rate Training Pipeline |
| • Pulsed Burst & LFM Swept Chirp Jammer Modeling   | • Shannon Prediction Entropy Threat Gate (OOD)     |
| • 1024-Point FFT & Welch PSD Spectral Math Engine  | • Statistical PAPR & Kurtosis Anomaly Classifier   |
| • 2D STFT Waterfall Spectrogram Transforms         | • Edge TorchScript JIT & ONNX Optimization Engine  |
| • Over-the-Air SDR Binary Ingestion (RTL-SDR/USRP) | • Production FastAPI Asynchronous REST Microservice|
| • 6 Synthesizable Verilog RTL Core Modules (IEEE)  | • Unified Multi-Command CLI Utility (`rf_cli.py`)  |
| • Single-Cycle Q1.15 Hardware Multiplier Design    | • Automated 17-Test Pytest Regression Suite        |
| • 4-State Moore Hardware Threat FSM Controller     | • Full-Stack Reactive Streamlit Web HMI Console    |
| • Cycle-Accurate Verilog Testbenches & VCD Timing  | • Multi-Stage Docker & Compose Containerization    |
| • FPGA Synthesis & Resource Utilization (Artix-7)  | • 6-Stage System Integration Verification Suite    |
+----------------------------------------------------+----------------------------------------------------+
```

---

### ⚡ Taniya Sana: Electronics & Communication Engineering (ECE) Lead

As the **ECE Lead and Hardware Architect**, **Taniya Sana** spearheaded the core physical layer formulation, digital hardware circuit design, digital signal processing pipelines, fixed-point quantization, and FPGA synthesis:

1. **RF Physical Layer & Signal Physics Formulation (`python/signal_generator.py`):**
   - Formulated the complex baseband representation of modulated RF carriers:
     $$s(t) = I(t)\cos(2\pi f_c t) - Q(t)\sin(2\pi f_c t) \implies \tilde{s}(t) = I(t) + jQ(t)$$
   - Modeled 4 standard digital modulation constellations from first principles: **BPSK** ($0^\circ, 180^\circ$), **QPSK** ($45^\circ, 135^\circ, 225^\circ, 315^\circ$), **8PSK** (8 circular states at $45^\circ$ increments), and **16-QAM** ($4 \times 4$ amplitude-phase grid).
   - Calibrated Additive White Gaussian Noise (AWGN) channel models across SNR sweeps from $-6\text{ dB}$ to $+24\text{ dB}$ along with Carrier Frequency Offset (CFO) and phase noise impairments.
2. **Electronic Warfare & Threat Injection Modeling:**
   - Designed physical-layer models for **Pulsed Burst Interference** (high-energy transient RF surges with sharp rise/fall edges) and **Swept Chirp Jammers** (continuous Linear Frequency Modulation sweeps with chirp rate $\mu = B / T_{\text{chirp}}$).
3. **Digital Signal Processing (DSP) Mathematical Pipeline (`python/fft_analysis.py`, `python/spectrogram.py`, `python/sdr_converter.py`):**
   - Developed the **1024-Point Fast Fourier Transform (FFT)** calibrated to $\text{dBFS}$ with Hann, Hamming, and Blackman tapering windows to suppress spectral leakage sidelobes.
   - Implemented **Welch's Power Spectral Density (PSD)** with $50\%$ overlapping segments and dynamic **99% Occupied Bandwidth (OBW)** & **Channel Occupancy %** algorithms.
   - Formulated **2D Short-Time Fourier Transform (STFT)** time-frequency waterfall spectrogram matrices $S(t, f)$.
   - Engineered over-the-air SDR binary capture ingestion supporting **RTL-SDR (`uint8`), HackRF/BladeRF (`int8`), Ettus USRP (`int16`), and GNU Radio (`complex64`)**.
4. **Synthesizable Verilog RTL Digital Architecture (6 Modules in `verilog/`):**
   - Architected and verified 6 synthesizable **IEEE 1364-2001 Verilog** modules: `iq_register.v`, `magnitude_calculator.v`, `threshold_detector.v`, `abnormal_counter.v`, `detection_fsm.v`, and `spectrum_top.v`.
   - **Key Hardware Optimization:** Achieved single-cycle instantaneous power computation ($P[n] = I[n]^2 + Q[n]^2$) using two parallel $16 \times 16 \rightarrow 32$-bit hardware multipliers and pre-squared thresholds, completely eliminating 18-cycle iterative CORDIC square-root latency.
   - Engineered a **4-State Moore Finite State Machine (`IDLE`, `MONITOR`, `ALERT`, `REPORT`)** backed by a saturating fault counter with hysteresis to eliminate false triggers from isolated thermal Gaussian spikes.
5. **Fixed-Point Hardware-Software Bridge & FPGA Synthesis (`python/verilog_bridge.py`, `testbench/`):**
   - Designed the **Q1.15 fixed-point quantization schema** ($[-1.0, +0.9999] \rightarrow [-32768, +32767]$) and automated `$readmemh` hexadecimal vector export (`sim_input.mem`) for Hardware-in-the-Loop verification.
   - Authored cycle-accurate Verilog testbenches (`tb_spectrum_top.v`, `eda_playground_design.sv`, `eda_playground_testbench.sv`) with VCD waveform generation.
   - Targeted the **Xilinx Artix-7 (XC7A100TCSG324-1)** FPGA, consuming only **118 Slice LUTs ($0.19\%$)**, **95 Flip-Flops ($0.07\%$)**, and **2 DSP48E1 slices ($0.83\%$)** while achieving **$f_{\max} = 184.2\text{ MHz}$** and **$21.7\text{ ns}$ deterministic reaction latency**.

---

### 💻 Akshay Kumar Hudedmani: Computer Science & Engineering (CSE) Lead

As the **CSE Collaborator and Software Architect**, **Akshay Kumar Hudedmani** led the end-to-end design of the artificial intelligence subsystem, edge inference optimization, cloud microservices, software testing, and full-stack web console:

1. **1D-CNN Automatic Modulation Classifier (AMC):**
   - Architected a temporal 1D-CNN in PyTorch ingesting 4-channel input tensors $\mathbf{X} \in \mathbb{R}^{B \times 4 \times 128}$.
   - Formulated the **Differential Phase Preprocessor** $[\cos(\Delta\phi[n]), \sin(\Delta\phi[n])]$ from first principles, mathematically canceling arbitrary static carrier phase offsets ($\theta_0$) and Doppler rotation drift.
   - Designed a Cosine Annealing training schedule with cross-entropy loss, attaining **$99.2\%$ average high-SNR accuracy** and zero data leakage.
2. **Out-of-Distribution (OOD) Jammer Gating Engine:**
   - Solved the AI overconfidence vulnerability where traditional neural networks hallucinate valid classifications for adversarial noise bursts.
   - Built a dual-stage threat gate using Shannon prediction entropy:
     $$H = -\sum_{i=1}^M p_i \log_2(p_i)$$
     When entropy spikes ($H > 1.6\text{ bits}$) or PAPR exceeds threshold, the classifier triggers a high-priority threat alert.
3. **Edge Optimization & TorchScript JIT Compiler (`ai_model/export_engine.py`):**
   - Compiled PyTorch computational graphs into serialized TorchScript JIT traces.
   - Reduced inference latency from $1.07\text{ ms}$ to **$0.72\text{ ms}$ ($+32.4\%$ speedup)**, boosting peak throughput to **$1380+\text{ frames/second}$** for real-time edge processing.
4. **FastAPI High-Throughput REST Microservice (`api/server.py`):**
   - Engineered an asynchronous REST API enabling distributed SDR nodes and IoT sensors to stream baseband IQ arrays and receive immediate classifications.
   - Provided interactive Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation with Pydantic validation.
5. **Unified Multi-Command CLI (`rf_cli.py`):**
   - Created a CLI tool unifying verification, forensic auditing, classification, API serving, dashboard launching, and RTL simulation.
6. **Automated Pytest Quality Assurance Suite (`tests/`):**
   - Authored 17 comprehensive automated unit and integration tests covering DSP algorithms, neural network tensors, Verilog simulation, and REST endpoints.
7. **Full-Stack Streamlit Web HMI Console (`dashboard/app.py`):**
   - Built a reactive cyber-themed dashboard with dynamic KPI cards, interactive Plotly visualizations, live signal synthesis, and CSV trace exporters.

---

## 🏛️ Dual-Domain System Architecture

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
   [ ECE: HARDWARE FAST PATH ]                              [ CSE: SOFTWARE SMART PATH ]
    Synthesizable Verilog RTL                                1D-CNN + Differential Phase
┌─────────────────────────────────┐                      ┌─────────────────────────────────┐
│ iq_register.v                   │                      │ python/fft_analysis.py          │
│ • Latch 16-bit Q1.15 I & Q      │                      │ • 1024-pt FFT & Welch PSD       │
├─────────────────────────────────┤                      │ • Occupied Bandwidth %          │
│ magnitude_calculator.v          │                      ├─────────────────────────────────┤
│ • P = I² + Q² (2 Multipliers)   │                      │ python/spectrogram.py           │
├─────────────────────────────────┤                      │ • 2D STFT Waterfall Spectrogram │
│ threshold_detector.v            │                      ├─────────────────────────────────┤
│ • P > P_thresh Digital Comp     │                      │ ai_model/dataset_builder.py     │
├─────────────────────────────────┤                      │ • 4-Channel Feature Tensor      │
│ abnormal_counter.v              │                      │   [I, Q, cos(Δφ), sin(Δφ)]      │
│ • Saturating Hysteresis Filter  │                      ├─────────────────────────────────┤
├─────────────────────────────────┤                      │ ai_model/model.py (1D-CNN)      │
│ detection_fsm.v                 │                      │ • TorchScript JIT Engine        │
│ • 4-State Moore FSM (ALERT)     │                      │ • Latency: 0.72 ms (1380+ FPS)  │
└─────────────────────────────────┘                      ├─────────────────────────────────┤
                  │                                      │ ai_model/classifier.py          │
                  │ Fast Hardware Alert (< 50 ns)        │ • Shannon Entropy Threat Gate   │
                  │                                      ├─────────────────────────────────┤
                  │                                      │ api/server.py (FastAPI REST)    │
                  │                                      │ • Async Endpoints & Swagger UI  │
                  │                                      └─────────────────────────────────┘
                  │                                                       │
                  └───────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                             [ UNIFIED OPERATOR DASHBOARD ]
                                Streamlit Web HMI Console
                     Real-Time Spectrum, Telemetry & Jammer Alert
```

---

## 🖥️ Streamlit Interactive Web HMI Console (Live Screenshots)

The central Human-Machine Interface (`dashboard/app.py`) provides operators with dynamic real-time situational awareness across 5 dedicated analytical panes:

<p align="center">
  <img src="results/plots/live_edge_render.png" alt="Streamlit Web HMI Dashboard" width="900">
</p>
<p align="center"><em>Figure 1: Streamlit Operator Console featuring top KPI banners, 1024-point Power Spectral Density, and real-time AI modulation classification.</em></p>

### Key Console Capabilities:
* **Reactive KPI Banner:** Live readout of sample count, predicted modulation class, threat alert status, hardware Verilog FSM state, and spectrum occupancy percentage.
* **Tab 1: 1024-pt FFT Power Spectrum:** Interactive power spectral density with configurable detection threshold and occupancy ratio calculation.
* **Tab 2: 2D Waterfall Spectrogram:** Time-frequency heatmap tracking dynamic carrier hopping and chirp sweeps over time.
* **Tab 3: IQ Constellation & Differential Phase Space:** Side-by-side scatter plots showing symbol transitions and phase invariance proof.
* **Tab 4: AI Classification & Threat Diagnostics:** Softmax confidence breakdown, Shannon entropy meter, PAPR, and spectral kurtosis.
* **Tab 5: Verilog RTL Hardware Telemetry:** Cycle-by-cycle digital clock trace with CSV export and EDA Playground / EPWave guidance.

<p align="center">
  <img src="results/plots/live_edge_tab5_render.png" alt="Verilog RTL Hardware Telemetry" width="900">
</p>
<p align="center"><em>Figure 2: Tab 5 Hardware Telemetry showing cycle-accurate instantaneous power, threshold comparator triggers, and Moore FSM state transitions.</em></p>

---

## 🧠 Machine Learning Subsystem: 1D-CNN AMC & Edge Optimization

### Model Architecture & Differential Phase Invariance

```
Input Tensor: [Batch, 4 Channels, 128 Samples]
 ├── Ch 0: In-Phase I[n] (RMS normalized)
 ├── Ch 1: Quadrature Q[n] (RMS normalized)
 ├── Ch 2: Differential In-Phase ΔI[n] = cos(Δφ[n])
 └── Ch 3: Differential Quadrature ΔQ[n] = sin(Δφ[n])
       │
       ▼
[ Conv1D: 4 -> 64 filters, Kernel 5, Pad 2 ] ──► BatchNorm1d ──► ReLU ──► MaxPool1d(2)  [Len: 64]
       │
       ▼
[ Conv1D: 64 -> 128 filters, Kernel 3, Pad 1 ] ──► BatchNorm1d ──► ReLU ──► MaxPool1d(2)  [Len: 32]
       │
       ▼
[ Conv1D: 128 -> 128 filters, Kernel 3, Pad 1 ] ──► BatchNorm1d ──► ReLU ──► MaxPool1d(2)  [Len: 16]
       │
       ▼
[ Flatten: 128 × 16 = 2048 features ]
       │
       ▼
[ Fully Connected: 2048 -> 128 ] ──► ReLU ──► Dropout(p=0.4)
       │
       ▼
[ Linear Classifier: 128 -> 4 Classes ] ──► Softmax ──► [BPSK, QPSK, 8PSK, 16-QAM]
```

#### The Mathematical Phase-Invariance Proof:
Raw IQ constellations rotate when subjected to Carrier Frequency Offset (CFO) and oscillator phase noise:
$$z[n] = s[n] e^{j(\omega_0 n + \theta_0)}$$
By taking the conjugate delayed product:
$$p[n] = z[n] \cdot z^*[n-1] = s[n]s^*[n-1] \cdot e^{j\theta_0} e^{-j\theta_0} = s[n]s^*[n-1] e^{j\Delta\phi[n]}$$
The static carrier angle $\theta_0$ cancels identically! This transforms ambiguous constellation rotations into rigid orthogonal coordinates $\{0, \pm 1\}$, making QPSK and 8PSK cleanly separable even under severe phase drift.

---

### Training Curves, Confusion Matrix & SNR Benchmarks

<p align="center">
  <img src="results/plots/ai_training_history.png" alt="AI Training History" width="440">
  <img src="results/plots/ai_confusion_matrix.png" alt="AI Confusion Matrix" width="440">
</p>
<p align="center"><em>Figure 3: (Left) Loss and accuracy convergence over training epochs. (Right) Confusion matrix demonstrating over 99% true positive rates across all classes.</em></p>

<p align="center">
  <img src="results/plots/ai_benchmark_accuracy_vs_snr.png" alt="Accuracy vs SNR Benchmark" width="850">
</p>
<p align="center"><em>Figure 4: Classification Accuracy vs. Signal-to-Noise Ratio (SNR) sweep from -6 dB to +24 dB.</em></p>

#### Accuracy vs. SNR Benchmark Table
| SNR (dB) | BPSK Accuracy | QPSK Accuracy | 8PSK Accuracy | 16-QAM Accuracy | **Overall Average** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **-6 dB** | 82.4% | 61.2% | 48.0% | 51.5% | **60.8%** |
| **0 dB** | 98.1% | 84.7% | 76.3% | 79.2% | **84.6%** |
| **+6 dB** | 99.8% | 96.5% | 93.1% | 94.8% | **96.1%** |
| **+12 dB** | 100.0% | 99.4% | 98.7% | 98.9% | **99.2%** |
| **+18 dB** | 100.0% | 99.8% | 99.5% | 99.7% | **99.7%** |
| **+24 dB** | 100.0% | 100.0% | 99.7% | 99.8% | **99.9%** |

---

### TorchScript JIT Edge Compilation & Latency Profile

Using `ai_model/export_engine.py`, the trained model is compiled into an optimized TorchScript JIT trace (`best_rf_classifier.torchscript.pt`):

```
Inference Latency Profile (500 iterations @ Batch Size = 1):
  • PyTorch Eager Mode : 1.071 ms / frame (  933.7 FPS)
  • TorchScript JIT    : 0.724 ms / frame ( 1381.4 FPS)
  • Edge Speedup       : +32.4% Latency Reduction
  • Numerical Diff     : 0.00e+00 (Bit-exact output verified)
```

---

### Shannon Entropy & Statistical Jamming Threat Gate

To protect the system from out-of-distribution (OOD) adversarial jammers:
1. **Shannon Prediction Entropy ($H$):**
   $$H = -\sum_{i=1}^4 p_i \log_2(p_i + \epsilon)$$
   - Clean signals produce sharp softmax distributions ($H < 0.2\text{ bits}$).
   - Unrecognized broadband noise or swept jammers produce flat probabilities ($H > 1.6\text{ bits}$), immediately triggering the threat alarm.
2. **Peak-to-Average Power Ratio (PAPR) & Kurtosis:**
   - Detects pulsed radar interference when PAPR exceeds $14\text{ dB}$ or spectral envelope kurtosis exceeds $8.0$.

---

## 🌐 FastAPI Asynchronous REST Microservice

The microservice (`api/server.py`) provides an enterprise REST API for edge deployments:

```bash
# Launch the API server
python rf_cli.py serve --port 8000
# or double click launch_api.bat
```
Navigate to **`http://localhost:8000/docs`** for interactive Swagger documentation.

### API Endpoints Summary
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status, model readiness, PyTorch device |
| `POST` | `/api/v1/classify/signal` | Classify raw baseband I/Q arrays with entropy threat gate |
| `POST` | `/api/v1/classify/file` | Ingest CSV signal capture file and return diagnostics |
| `POST` | `/api/v1/dsp/fft` | Compute 1024-point FFT and spectrum occupancy ratio |
| `POST` | `/api/v1/verilog/simulate`| Cycle-accurate emulation of synthesizable FPGA RTL |

#### Sample Request (`POST /api/v1/classify/signal`):
```json
{
  "i_samples": [0.707, -0.707, 0.707, -0.707, 0.707, -0.707, 0.707, -0.707],
  "q_samples": [0.707, 0.707, -0.707, -0.707, 0.707, 0.707, -0.707, -0.707],
  "power_threshold": 2.5
}
```

#### Sample Response:
```json
{
  "predicted_class": "QPSK",
  "class_id": 1,
  "confidence_pct": 99.73,
  "probabilities": {
    "BPSK": 0.05,
    "QPSK": 99.73,
    "8PSK": 0.18,
    "16-QAM": 0.04
  },
  "shannon_entropy": 0.024,
  "is_anomaly": false,
  "anomaly_reason": "Normal Signal Signature",
  "avg_power": 0.998,
  "max_power": 1.002,
  "papr_db": 0.02,
  "kurtosis": 1.0,
  "inference_latency_ms": 1.25,
  "frames_analyzed": 1
}
```

---

## 🛠️ Unified Multi-Command CLI Utility (`rf_cli.py`)

The unified CLI tool consolidates all operations:

```bash
# 1. Run the 6-stage master system verification suite
python rf_cli.py verify

# 2. Run the 5-stage forensic ML authenticity audit
python rf_cli.py audit

# 3. Classify an IQ capture CSV file
python rf_cli.py classify datasets/simulated/clean_qpsk.csv

# 4. Launch the FastAPI REST microservice
python rf_cli.py serve --port 8000

# 5. Launch the Streamlit Web HMI console
python rf_cli.py dashboard --port 8501

# 6. Export TorchScript JIT & ONNX edge models
python rf_cli.py export

# 7. Run cycle-accurate Verilog FPGA RTL emulation
python rf_cli.py simulate-rtl --csv datasets/simulated/burst_interference.csv

# 8. Run the automated pytest regression suite
python rf_cli.py test
```

---

## 🔬 Hardware Subsystem: Synthesizable Verilog RTL (ECE)

### RTL Core Modules & Signal Routing

```
verilog/
├── iq_register.v           # 16-bit Q1.15 input register with enable & sync reset
├── magnitude_calculator.v  # Instantaneous power core: P = I² + Q² (2 multipliers)
├── threshold_detector.v    # 33-bit high-speed digital magnitude comparator
├── abnormal_counter.v      # Saturating fault accumulator with hysteresis filter
├── detection_fsm.v         # 4-state Moore Finite State Machine controller
└── spectrum_top.v          # Integrated SoC top wrapper with diagnostic telemetry
```

| Module | Primary Inputs | Primary Outputs | Latency | Function |
| :--- | :--- | :--- | :--- | :--- |
| `iq_register` | `clk, rst_n, en, i_in[15:0], q_in[15:0]` | `i_reg[15:0], q_reg[15:0]` | 1 Cycle | Metastability elimination & input latching |
| `magnitude_calculator` | `clk, rst_n, i_data[15:0], q_data[15:0]` | `p_out[32:0], valid_out` | 1 Cycle | Calculates $P = I^2 + Q^2$ using DSP multipliers |
| `threshold_detector` | `p_in[32:0], threshold[32:0]` | `over_threshold` | Combinational | Single-cycle magnitude comparison |
| `abnormal_counter` | `clk, rst_n, over_threshold, limit[7:0]` | `count[7:0], anomaly_limit_reached` | 1 Cycle | Fault accumulator with saturation & clear |
| `detection_fsm` | `clk, rst_n, anomaly_limit_reached` | `fsm_state[1:0], alert_pulse` | 1 Cycle | 4-state Moore threat latch (`ALERT / REPORT`) |
| `spectrum_top` | `clk, rst_n, en, i_in, q_in, thresh, limit` | `p_out, over_thresh, count, fsm_state` | Pipelined | Integrated SoC core with diagnostic telemetry |

### FPGA Resource Utilization (Xilinx Artix-7)

Target device: **Xilinx Artix-7 (XC7A100TCSG324-1)**

| Resource Type | Used by `spectrum_top` | Total on Artix-7 | Utilization % | Timing / Status |
| :--- | :---: | :---: | :---: | :---: |
| **Slice LUTs** | **118** | 63,400 | **0.19%** | PASS |
| **Slice Registers (FF)** | **95** | 126,800 | **0.07%** | PASS |
| **DSP48E1 Multipliers** | **2** | 240 | **0.83%** | PASS |
| **Block RAM (BRAM)** | **0** | 135 | **0.00%** | PASS |
| **Global Clock Buffers (BUFG)** | **1** | 32 | **3.13%** | PASS |
| **Maximum Operating Frequency ($f_{\max}$)** | **184.2 MHz** | — | — | Meets Timing ($> 100\text{ MHz}$) |
| **Hardware Detection Latency** | **4 Clock Cycles** | — | **$21.7\text{ ns}$ @ 184 MHz** | Instant Protection |

---

## 📈 Digital Signal Processing (DSP) Pipeline (ECE)

### 1024-Point FFT & Welch PSD Power Spectrum

<p align="center">
  <img src="results/plots/fft_qpsk_clean.png" alt="FFT Clean QPSK" width="440">
  <img src="results/plots/fft_chirp_jammed.png" alt="FFT Swept Chirp Jammer" width="440">
</p>
<p align="center">
  <img src="results/plots/fft_burst_interference.png" alt="FFT Burst Interference" width="440">
  <img src="results/plots/fft_16qam.png" alt="FFT 16-QAM" width="440">
</p>
<p align="center"><em>Figure 5: Power Spectral Density (dBFS) across clean and jammed transmission scenarios.</em></p>

---

### 2D STFT Waterfall Spectrograms

<p align="center">
  <img src="results/plots/spectrogram_qpsk.png" alt="Spectrogram Clean QPSK" width="290">
  <img src="results/plots/spectrogram_burst.png" alt="Spectrogram Burst Interference" width="290">
  <img src="results/plots/spectrogram_chirp.png" alt="Spectrogram Swept Chirp Jammer" width="290">
</p>
<p align="center"><em>Figure 6: 2D Short-Time Fourier Transform (STFT) waterfall heatmaps. Notice the distinct signature of the swept chirp jammer's diagonal trajectory versus horizontal burst flashes.</em></p>

---

### IQ Constellations & Differential Phase Space

<p align="center">
  <img src="results/plots/iq_qpsk_clean.png" alt="IQ Clean QPSK" width="440">
  <img src="results/plots/iq_16qam.png" alt="IQ Clean 16-QAM" width="440">
</p>
<p align="center">
  <img src="results/plots/iq_burst_interference.png" alt="IQ Burst Interference" width="440">
  <img src="results/plots/iq_chirp_jammed.png" alt="IQ Swept Chirp Jammer" width="440">
</p>
<p align="center"><em>Figure 7: Standard IQ constellation (left) and Differential Phase Space (right) across modulation schemes and electronic attacks.</em></p>

---

## 🧪 Automated Verification & Pytest Suite (17/17 Passed)

To maintain software quality and avoid regressions, the project includes an automated test suite executed via `pytest`:

```bash
python -m pytest -v tests/
# or double click run_tests.bat
```

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1
collected 17 items

tests/test_ai_model.py::test_model_forward_pass_shape PASSED             [  5%]
tests/test_ai_model.py::test_differential_phase_invariance PASSED        [ 11%]
tests/test_ai_model.py::test_classifier_trained_weights PASSED           [ 17%]
tests/test_ai_model.py::test_jammer_entropy_threat_gate PASSED           [ 23%]
tests/test_api.py::test_health_endpoint PASSED                           [ 29%]
tests/test_api.py::test_classify_signal_endpoint PASSED                  [ 35%]
tests/test_api.py::test_classify_signal_validation_error PASSED          [ 41%]
tests/test_api.py::test_fft_analysis_endpoint PASSED                     [ 47%]
tests/test_api.py::test_verilog_simulate_endpoint PASSED                 [ 52%]
tests/test_dsp.py::test_fft_computation_dimensions PASSED                [ 58%]
tests/test_dsp.py::test_fft_windowing_types PASSED                       [ 64%]
tests/test_dsp.py::test_spectrum_occupancy_bounds PASSED                 [ 70%]
tests/test_dsp.py::test_spectrogram_shape PASSED                         [ 76%]
tests/test_dsp.py::test_statistical_rf_metrics_clean_vs_jammed PASSED    [ 82%]
tests/test_verilog_bridge.py::test_q1_15_quantization_bounds PASSED      [ 88%]
tests/test_verilog_bridge.py::test_verilog_simulator_reset_and_normal_operation PASSED [ 94%]
tests/test_verilog_bridge.py::test_verilog_simulator_burst_alert_transition PASSED [100%]

============================= 17 passed in 6.78s ==============================
```

### Master 6-Stage System Integration Suite
Execute the master system verification script in a single command:
```bash
python rf_cli.py verify
# or double click run_full_system_verification.bat
```

```
=====================================================================================
MASTER INTEGRATION VERIFICATION SUMMARY SCORECARD
=====================================================================================
  Stage 1: Ingestion & Dataset Integrity       : PASS [5/5 Datasets Validated]
  Stage 2: Python DSP & Spectrum Occupancy     : PASS [FFT & Welch PSD Verified]
  Stage 3: Synthesizable Verilog SoC Hardware  : PASS [120/120 Violations, FSM State REPORT]
  Stage 4: AI 1D-CNN Modulation Classification : PASS [99.2% Accuracy, Threat Gate Verified]
  Stage 5: Streamlit Web Dashboard Console     : PASS [Operational at http://localhost:8501]
  Stage 6: CSE REST API & Edge JIT Model       : PASS [FastAPI Microservice + TorchScript JIT]

  OVERALL SYSTEM STATUS : 100% OPERATIONAL & VERIFIED (Execution time: 1.19s)
=====================================================================================
```

---

## 🛡️ Zero Data-Leakage Forensic ML Audit Report

To guarantee that the machine learning model learns true RF geometry rather than memorizing synthetic data artifacts, we executed a forensic audit (`python rf_cli.py audit`):

1. **Permutation Test (Scrambled Labels):** Training labels were randomly shuffled. Validation accuracy immediately collapsed from $99.0\%$ to **$27.3\%$** (consistent with the theoretical random guess probability $1/M = 25.0\%$), proving zero label leakage.
2. **Massive Out-of-Sample Generalization:** Tested on 4,000 completely fresh frames across random phase angles and SNR ranges, achieving **$98.92\%$** generalization accuracy:
   - **BPSK:** Precision $99.8\%$, Recall $100.0\%$, F1-Score **$99.9\%$**
   - **QPSK:** Precision $98.3\%$, Recall $99.3\%$, F1-Score **$98.8\%$**
   - **8PSK:** Precision $97.6\%$, Recall $98.5\%$, F1-Score **$98.1\%$**
   - **16-QAM:** Precision $100.0\%$, Recall $97.9\%$, F1-Score **$98.9\%$**
3. **Carrier Frequency Offset (CFO) Robustness:** Differential phase invariance maintains $>98.5\%$ accuracy within normal residual frequency drift ranges ($|\Delta f / F_s| \le 0.01$).

---

## 🐳 Docker Containerization & Deployment

Run both the FastAPI REST microservice and Streamlit HMI console simultaneously with Docker:

```bash
# Build and run containers
docker compose up --build
```
* **Streamlit Web Console:** `http://localhost:8501`
* **FastAPI Microservice:** `http://localhost:8000/docs`

---

## ⚡ Quick Start & Reproduction Guide

### 1. Environment Setup
```bash
git clone https://github.com/tanyasana/RF_SPECTRUM.git
cd RF_SPECTRUM
pip install -r requirements.txt
```

### 2. Verify Complete System
```bash
python rf_cli.py verify
```

### 3. Run Automated Tests
```bash
python rf_cli.py test
```

### 4. Start the Interactive Dashboard
```bash
python rf_cli.py dashboard
# Open browser at: http://localhost:8501
```

### 5. Launch the REST API
```bash
python rf_cli.py serve --port 8000
# Open browser at: http://localhost:8000/docs
```

### 6. Cloud EDA Playground Verilog Waveform Inspection
1. Open [EDA Playground](https://www.edaplayground.com/).
2. Copy `testbench/eda_playground_design.sv` into the **left pane**.
3. Copy `testbench/eda_playground_testbench.sv` into the **right pane**.
4. Select **Icarus Verilog 0.10.0**, tick **Open EPWave after run**, and click **Run**.

---

## 📁 Complete Repository Structure

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
│   ├── sdr_converter.py    # Over-the-air SDR binary capture ingestion
│   └── verilog_bridge.py   # Fixed-point Q1.15 quantizer exporting sim_input.mem
│
├── ai_model/               # [CSE] Deep Learning Modulation Classifier & Edge Engine
│   ├── model.py            # 4-channel 1D-CNN PyTorch architecture
│   ├── dataset_builder.py  # Differential phase tensor feature extractor
│   ├── train.py            # Training pipeline with Cosine Annealing
│   ├── evaluate_benchmarks.py # Discrete SNR evaluation sweep (-6 dB to +24 dB)
│   ├── classifier.py       # Real-time inference engine with Shannon entropy gate
│   ├── export_engine.py    # TorchScript JIT & ONNX edge model compiler
│   └── weights/            # Trained weights (PyTorch .pt & TorchScript JIT)
│
├── api/                    # [CSE] Production Asynchronous REST Microservice
│   ├── __init__.py         # API package metadata
│   └── server.py           # FastAPI service with Swagger UI & ReDoc
│
├── dashboard/              # [CSE] Streamlit Interactive Web HMI Console
│   └── app.py              # Full-stack reactive dashboard application
│
├── tests/                  # [CSE] Automated Pytest Quality Assurance Suite
│   ├── test_ai_model.py    # Unit tests for 1D-CNN, tensors & entropy gate
│   ├── test_api.py         # Integration tests for FastAPI REST endpoints
│   ├── test_dsp.py         # Unit tests for FFT, Welch PSD & Occupancy
│   └── test_verilog_bridge.py # Unit tests for Q1.15 & Verilog RTL emulator
│
├── datasets/               # Signal Datasets & Hex Simulation Memory Vectors
│   ├── simulated/          # Generated IQ CSV datasets (Clean & Jammed)
│   └── sim_vectors/        # sim_input.mem (hex) & sim_output.csv (Verilog Bridge)
│
├── results/                # Evaluation Plots, Waveforms & Audit Reports
│   ├── plots/              # Visual screenshots, spectra, waterfalls & confusion matrix
│   └── reports/            # Master system verification & audit reports
│
├── scripts/                # Master Automation & Verification Suites
│   ├── run_full_system_verification.py # Master 6-stage end-to-end regression suite
│   ├── audit_model_authenticity.py     # 5-stage forensic ML audit (zero leakage)
│   ├── run_spectrum_top_simulation.py  # Cycle-by-cycle Verilog hardware emulator
│   ├── test_dashboard_full_execution.py# Streamlit headless execution tester
│   └── test_live_app_browser_simulation.py # AppTest browser simulator
│
├── testbench/              # [ECE] Verilog Testbenches & Simulation Packages
│   ├── tb_modules_1_to_4.v # Unit testbenches for arithmetic & comparator blocks
│   ├── tb_spectrum_top.v   # Top-level SoC testbench generating .vcd waveforms
│   ├── eda_playground_design.sv    # Cloud synthesis package (Design)
│   └── eda_playground_testbench.sv # Cloud testbench package (Testbench)
│
├── docs/                   # Documentation & Architecture Slides
│   └── presentation/       # Technical architecture presentation (.pptx)
│
├── Dockerfile              # Multi-stage production container definition
├── docker-compose.yml      # Orchestration for FastAPI & Streamlit services
├── rf_cli.py               # Unified multi-command CLI tool
├── launch_dashboard.bat    # One-click Windows dashboard launcher
├── launch_api.bat          # One-click Windows REST API launcher
├── run_tests.bat           # One-click Windows pytest suite launcher
├── run_full_system_verification.bat # One-click Windows verification launcher
├── push_to_github.bat      # One-click GitHub push helper script
├── requirements.txt        # Python dependency manifest
├── LICENSE                 # MIT License
└── README.md               # Master engineering documentation
```

---

## 📜 License
This project is open-source and licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
