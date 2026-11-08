# Capstone Defense & Technical Interview (Viva Voce) Preparation Guide

**Project Title:** AI-Driven RF Signal Classification & Interference Detection System  
**Co-Design Domain:** Digital Signal Processing (DSP) + Synthesizable Verilog (DSD) + Deep Learning (AMC) + Web HMI (Streamlit)

---

## 🏛️ Category 1: Digital Signal Processing (DSP) & RF Theory

### Q1: What is IQ data and why is it used instead of real RF samples?
* **Answer:** Any modulated RF bandpass signal can be represented as:
  $$s(t) = I(t)\cos(2\pi f_c t) - Q(t)\sin(2\pi f_c t)$$
  In discrete baseband processing, we represent each sample as a complex number:
  $$x[n] = I[n] + j Q[n]$$
  Using IQ data:
  1. **Eliminates frequency ambiguity:** Real signals have symmetric positive and negative frequency spectra. Complex IQ signals allow two-sided spectra spanning $[-F_s/2, +F_s/2]$, enabling positive vs. negative frequency distinction.
  2. **Direct Envelope & Phase Recovery:** Instantaneous power is $P = I^2 + Q^2$ and instantaneous phase is $\theta = \text{atan2}(Q, I)$.

### Q2: Why did you use a Hann window before computing the 1024-point FFT?
* **Answer:** An FFT assumes the finite $N$-sample sequence repeats infinitely. If the beginning and end of the sample frame do not match continuously, an artificial step discontinuity occurs at the boundary, causing **spectral leakage** (energy spilling into adjacent bins). 
* Applying a Hann window $w[n] = 0.5 - 0.5\cos(2\pi n / N)$ smoothly attenuates the edges to zero, suppressing sidelobes from $-13\text{ dB}$ (rectangular) down to $-31\text{ dB}$ (Hann).

### Q3: How is dynamic Spectrum Occupancy calculated?
* **Answer:** 
  1. We compute the normalized logarithmic Power Spectral Density (PSD in dBFS):
     $$\text{PSD}[k] = 10 \log_{10}\left(\frac{|X[k]|^2}{N \cdot F_s}\right) - \max(\text{PSD})$$
  2. We count how many discrete frequency bins exceed the energy threshold (e.g., $P_{th} = -20\text{ dBFS}$):
     $$\text{Occupancy (\%)} = \left(\frac{N_{\text{bins}}(\text{PSD} > P_{th})}{N_{\text{total}}}\right) \times 100$$

---

## ⚡ Category 2: Synthesizable Verilog Digital System Design (DSD)

### Q4: Why did you compute Power as $P = I^2 + Q^2$ instead of Magnitude $\sqrt{I^2 + Q^2}$ in hardware?
* **Answer:** This is a classic **hardware-software co-design optimization**:
  * Computing $\sqrt{I^2 + Q^2}$ requires an iterative **CORDIC (Coordinate Rotation Digital Computer)** pipeline or a digit-by-digit square root engine. This consumes 16+ clock cycles of pipeline latency and hundreds of FPGA logic slices.
  * In contrast, comparing power against a pre-squared digital threshold ($P > P_{th}^2$) produces the **exact same physical detection decision** using only two 16x16-bit DSP multipliers and one 32-bit adder in a single clock cycle.

### Q5: What is the wordlength and fixed-point format used in the RTL?
* **Answer:** 16-bit signed **Q1.15 fixed-point format**:
  * 1 sign bit (Bit 15).
  * 15 fractional magnitude bits (Bits 14 down to 0).
  * Numerical range: $[-1.0, +0.999969]$.
  * Full-scale integer representation: $1.0 \equiv 32,768$ (or $16,384$ with $6\text{ dB}$ headroom).
  * When two 16-bit numbers are squared and added, the power product occupies up to 32 bits, which is handled by our 32-bit comparator without overflow.

### Q6: Explain the Detection FSM state machine. Why is it a Moore machine?
* **Answer:** It is a 4-state sequential controller:
  * `IDLE (2'b00)`: Reset state; registers cleared, alert low.
  * `MONITOR (2'b01)`: Normal reception; power is monitored continuously.
  * `ALERT (2'b10)`: Triggered when `abnormal_counter >= anomaly_limit` (5 consecutive violations). Asserts `alert_flag = 1`.
  * `REPORT (2'b11)`: Triggered after the operator/host CPU acknowledges `reset_alert = 1`. Asserts `report_ready = 1` and safely resets the violation counter before returning to `MONITOR`.
* It is designed as a **Moore machine** because the control outputs depend strictly on the current registered state, not asynchronously on the inputs. This prevents combinational glitches from propagating to the downstream SoC interrupt bus.

### Q7: What is the total pipeline latency of the 6 Verilog modules?
* **Answer:** **2 clock cycles** from raw sample arrival to threshold detection:
  * Cycle 1: `iq_register` samples and registers $I$ and $Q$.
  * Cycle 2: `magnitude_calculator` computes and registers $P = I^2 + Q^2$.
  * Cycle 3 (Combinational edge): `threshold_detector` compares $P > P_{th}$ and `abnormal_counter` increments synchronously on the rising edge.

---

## 🧠 Category 3: Deep Learning (1D-CNN AMC & Threat Gating)

### Q8: What was the root cause of the initial $-2.5\%$ gap in QPSK classification, and how did you solve it?
* **Answer:**
  * **The Root Cause:** In uncalibrated wireless receivers, an incoming packet has an unknown carrier phase offset $\theta_0 \sim \mathcal{U}(0, 2\pi)$. Because both QPSK and 8PSK lie on the same constant-modulus unit circle ($|s| = 1$), rotating QPSK by an arbitrary angle causes its 4 constellation points to land on 8PSK phase bins, confusing the network.
  * **The Solution:** We implemented **Differential Phase Preprocessing**:
    $$p[n] = r[n] \cdot r^*[n-1] = s[n]s^*[n-1]e^{j(\theta_0 - \theta_0)} = s[n]s^*[n-1]$$
    Carrier offset $\theta_0$ cancels out completely. By feeding $\Delta I = \cos(\Delta \phi)$ and $\Delta Q = \sin(\Delta \phi)$ alongside raw $I$ and $Q$ in a 4-channel tensor, QPSK phase jumps ($\Delta \phi \in \{0, \pm 90^\circ, 180^\circ\}$) became mathematically orthogonal to 8PSK jumps ($\Delta \phi \in \{0, \pm 45^\circ, \dots\}$).
  * **The Result:** High-SNR accuracy surged to **$99.2\%$ average** and **$99.8\%$ peak**.

### Q9: Why use a 1D-CNN instead of a 2D-CNN (like ResNet) or an LSTM?
* **Answer:**
  1. **Latency:** A 2D-CNN requires converting time-series into 2D spectrogram images, which introduces STFT window buffering latency ($>20\text{ ms}$). Our 1D-CNN operates directly on raw discrete samples in **$1.02\text{ ms}$**, meeting real-time SDR constraints.
  2. **Phase Preserving:** 2D spectrograms discard instantaneous phase information, capturing only spectral energy. 1D temporal convolutions preserve the fine-grained relative phase steps between adjacent samples.
  3. **No Recurrent Bottlenecks:** LSTMs/GRUs process samples sequentially ($O(N)$ sequential steps), which cannot be parallelized efficiently on edge hardware. 1D convolutions execute in parallel across all time steps.

### Q10: How did you prove the 99% benchmark is authentic and not overfitted or "fake hype"?
* **Answer:** We executed an independent **5-Stage Forensic ML Audit**:
  1. **Permutation Test:** When ground-truth labels were randomly scrambled during evaluation, accuracy collapsed from $99.4\%$ to **$25.1\%$** (the exact mathematical random guess floor of $1/4 = 25.0\%$). If data leakage existed, accuracy would have remained artificially high.
  2. **Massive Out-of-Sample Test:** Evaluated over 4,000 completely fresh frames with independent random seeds, accuracy was **$98.83\%$** (BPSK: 99.8%, QPSK: 98.2%, 8PSK: 97.8%, 16-QAM: 99.5%).
  3. **Stress Tests:** We systematically determined the model's physical boundaries: robust up to $\pm 1\%$ Carrier Frequency Offset ($\Delta f / F_s \le 0.01$) and tolerant to timing jitter in the intra-symbol eye region.

### Q11: How does the Hybrid Jamming Gate work?
* **Answer:** The system uses a two-pronged threat gate:
  1. **Physical Energy Gate:** If peak power exceeds $P_{limit} = 2.50$, it triggers a `High-Power Jamming Burst` alert (detecting pulsed radar / RF energy weapons).
  2. **Shannon Prediction Entropy Gate:** 
     $$H = -\sum_{i=1}^{M} P_i \log_2(P_i)$$
     When an adversarial jammer emits broadband noise or swept chirps that do not conform to any legitimate modulation alphabet, the softmax probability spreads evenly ($P \approx [0.25, 0.25, 0.25, 0.25]$), causing Shannon Entropy to spike ($H > 1.6\text{ bits}$). This flags an `Unrecognized / Threat Jammer Signature`.

---

## 💻 Category 4: System Integration & Web HMI (Streamlit)

### Q12: How are the hardware Verilog domain and Python AI domain linked together?
* **Answer:** Through the **Hardware-Software Co-Design Bridge** ([`python/verilog_bridge.py`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/python/verilog_bridge.py)):
  1. High-level floating-point RF samples are quantized into fixed-point 16-bit hex integers and exported to `datasets/sim_vectors/sim_input.mem`.
  2. The Verilog testbench ingests `sim_input.mem` using `$readmemh` and logs cycle-by-cycle register transitions to `sim_output.csv`.
  3. The Streamlit dashboard ingests both the software AI predictions and the hardware register telemetry, presenting them in a synchronized operator view.

### Q13: If this system were deployed on physical hardware, what would the architecture look like?
* **Answer:**
  * **RF Front-End:** An SDR receiver (e.g., AD9361 on a Xilinx Zynq UltraScale+ MPSoC or USRP).
  * **PL (Programmable Logic - FPGA):** Synthesizes our 6 Verilog modules (`spectrum_top.v`) clocked at $100\text{ MHz}$ to perform wire-speed $I^2+Q^2$ power computation, hardware filtering, and microsecond-level fault alerting.
  * **PS (Processing System - ARM Cortex-A53 / Edge CPU):** Ingests sample blocks via DMA (Direct Memory Access), executes the PyTorch 1D-CNN inference engine ($1.02\text{ ms}$ execution), and serves the Streamlit dashboard over a secure local gigabit network.
