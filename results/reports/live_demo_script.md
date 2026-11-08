# 3-Minute Live Demonstration Walkthrough Script

**Project Title:** AI-Driven RF Signal Classification & Interference Detection System  
**Audience:** Evaluation Committee, Academic Faculty, or Technical Defense Panel  
**Time Limit:** 3 to 5 Minutes  
**Tool Used:** Streamlit Web Dashboard ([`launch_dashboard.bat`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/launch_dashboard.bat) or [http://localhost:8501](http://localhost:8501))

---

## 🎬 Pre-Demo Setup (Before Starting)
1. Double-click [`launch_dashboard.bat`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/launch_dashboard.bat) or run:
   ```bash
   streamlit run dashboard/app.py
   ```
2. Open your web browser to **`http://localhost:8501`**.
3. Maximize the window (or press `F11` for full-screen presentation mode).

---

## ⏱️ Minute 1: The Problem & Baseline Clean Signal (QPSK)

### Action 1: Point to the Top Banner & Select Clean QPSK
* **Sidebar:** Select `Pre-Generated Datasets` $\rightarrow$ `Clean QPSK (Normal Transmission)`.
* **What to Say:**
  > *"Good morning committee. Today I am demonstrating our Hardware-Software Co-Design for real-time RF spectrum monitoring and automated threat detection. At the top of the dashboard, you see our live telemetry banners: active sample count, AI classification with confidence score, threat status badge, Verilog hardware FSM state, and monitored spectrum occupancy."*

### Action 2: Show Tab 1 (Spectral Analysis) & Tab 3 (Constellation)
* **Click Tab 1 (Spectral Analysis):**
  > *"Here in Tab 1, our 1024-point FFT centers the two-sided baseband from $-500\text{ kHz}$ to $+500\text{ kHz}$. Notice the clean QPSK main lobe occupying approximately $22.9\%$ of the monitored band above our $-20\text{ dBFS}$ threshold line."*
* **Click Tab 3 (IQ & Phase Space):**
  > *"In Tab 3, on the left is the raw IQ constellation with 4 distinct quadrants. On the right is our novel **Differential Phase Space ($\Delta Q$ vs. $\Delta I$)**. By computing the conjugate product $p[n] = z[n] \cdot z^*[n-1]$, the unknown carrier phase offset $\theta_0$ cancels out completely $(\theta_0 - \theta_0 = 0)$, organizing phase jumps into orthogonal coordinates $\{0, \pm 1\}$."*

---

## ⏱️ Minute 2: Threat Injection & Hardware Verilog Telemetry

### Action 3: Switch to Pulsed Burst Interference
* **Sidebar:** Select `Pulsed Burst Interference (Energy Surge)`.
* **What to Say:**
  > *"Now, let's simulate an active adversarial attack. I am selecting our high-power pulsed burst interference dataset."*

### Action 4: Point to the Immediate Threat Banners & Tab 5
* **Point to Top Banner:**
  > *"Notice how our system reacts instantly: the threat badge pulses red to **THREAT / JAMMER**, and our Verilog FSM advances to **ALERT / REPORT**."*
* **Click Tab 5 (Verilog Hardware Telemetry):**
  > *"This tab showcases our synthesizable Verilog hardware core running cycle-by-cycle. In Trace 1, our pipelined $P = I^2 + Q^2$ multiplier computes instantaneous power in just 1 clock cycle without expensive square roots. When power surges past the threshold line, Trace 2 flags the digital comparator hits, and Trace 3 accumulates the 5 consecutive violations, immediately transitioning our 4-state Moore FSM to ALERT."*

---

## ⏱️ Minute 3: Waterfall Spectrogram & Deep Learning AI Gating

### Action 5: Switch to Swept Chirp Jammer & View Tab 2
* **Sidebar:** Select `Swept Chirp Jammer (Broadband Threat)`.
* **Click Tab 2 (2D Waterfall Spectrogram):**
  > *"Next, we test a dynamic broadband threat: a swept frequency chirp jammer. A standard FFT collapses time and misses the trajectory. But in our 2D Short-Time Fourier Transform Waterfall Spectrogram, you clearly see the bright diagonal trajectory as the jammer sweeps across baseband channels over time."*

### Action 6: View Tab 4 (AI Modulation Classifier & Threat Diagnostics)
* **Click Tab 4 (AI Modulation Classifier):**
  > *"Finally, in Tab 4, our 1D-CNN deep learning classifier runs in just $1.02\text{ ms}$. For normal signals, it achieves **$99.2\%$ average accuracy**. But for this non-standard swept chirp threat, our **Shannon Prediction Entropy** spikes to $1.10\text{ bits}$, triggering our hybrid threat gate to flag an unrecognized adversarial signature."*

---

## 🎯 Concluding Summary (Last 15 Seconds)
* **What to Say:**
  > *"In conclusion, this project bridges the microsecond wire-speed alerting of an FPGA with the high-level cognitive classification of deep learning, all verified with zero data leakage and operating live on our dashboard. Thank you, and I am now ready for your questions."*
