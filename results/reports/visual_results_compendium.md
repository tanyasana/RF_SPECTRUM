# Visual Results & Figure Compendium

**Project:** AI-Driven RF Signal Classification & Interference Detection System  
**Artifact Directory:** [`results/plots/`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/)  
**Total High-Resolution Figures:** 16 Generated Figures

---

## 1. Deep Learning Model Training & Benchmark Figures

### Figure 1: Accuracy vs. Signal-to-Noise Ratio (SNR) Benchmark Curve
* **File:** [`ai_benchmark_accuracy_vs_snr.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/ai_benchmark_accuracy_vs_snr.png)
* **Description:** Evaluates classification accuracy from $-6\text{ dB}$ to $+24\text{ dB}$ in $3\text{ dB}$ discrete steps across 4,400 test frames. 
* **Key Finding:** Achieves **$99.2\%$ average accuracy** in the high-SNR regime ($\ge 12\text{ dB}$) and peaks at **$99.8\%$**, exceeding the theoretical $\ge 95.0\%$ target while collapsing gracefully to the $25.0\%$ random guess limit at $-6\text{ dB}$.

### Figure 2: AMC Validation Confusion Matrix (Differential Phase Preprocessed)
* **File:** [`ai_confusion_matrix.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/ai_confusion_matrix.png)
* **Description:** Normalized 4-class multi-classification confusion matrix across BPSK, QPSK, 8PSK, and 16-QAM.
* **Key Finding:** BPSK achieves $100.0\%$, 16-QAM achieves $99.0\%$, and QPSK achieves $98.5\%$, confirming that differential phase preprocessing ($\Delta I, \Delta Q$) completely resolves carrier phase ambiguity.

### Figure 3: Deep Learning Training & Validation History
* **File:** [`ai_training_history.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/ai_training_history.png)
* **Description:** Convergence trajectories across 18 epochs showing Cross-Entropy Loss (left) and Accuracy (right).
* **Key Finding:** Training loss drops to $0.0407$ while validation accuracy reaches $95.6\%$ across the entire mixed-SNR dataset with zero divergent overfitting.

---

## 2. Time-Frequency Waterfall Spectrograms (STFT)

### Figure 4: Swept-Frequency Chirp Jammer Waterfall
* **File:** [`spectrogram_chirp.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/spectrogram_chirp.png)
* **Description:** Short-Time Fourier Transform 2D heatmap showing a broadband swept chirp jammer.
* **Physical Insight:** Visualizes the clear linear diagonal trajectory as the jammer sweeps through channels across time ($df / dt \neq 0$).

### Figure 5: Pulsed Burst Interference Waterfall
* **File:** [`spectrogram_burst.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/spectrogram_burst.png)
* **Description:** STFT spectrogram of high-power pulsed energy spikes.
* **Physical Insight:** Captures sudden, high-intensity horizontal energy bands occurring intermittently across specific time intervals.

### Figure 6: Clean QPSK Baseline Waterfall
* **File:** [`spectrogram_qpsk.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/spectrogram_qpsk.png)
* **Description:** Baseline waterfall displaying steady, continuous channel occupancy.

---

## 3. Power Spectral Density (1024-Point FFT Analysis)

* **Clean QPSK Spectrum:** [`fft_qpsk_clean.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/fft_qpsk_clean.png) (Occupancy: $22.9\%$, centered at baseband).
* **Noisy BPSK Spectrum:** [`fft_bpsk_noisy.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/fft_bpsk_noisy.png) (Occupancy: $47.2\%$, showing elevated noise floor at $\text{SNR} = 6\text{ dB}$).
* **Clean 16-QAM Spectrum:** [`fft_16qam.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/fft_16qam.png) (Occupancy: $27.1\%$, compact roll-off).
* **Burst Jammer Spectrum:** [`fft_burst_interference.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/fft_burst_interference.png) (Massive energy surge breaching threshold line).
* **Swept Chirp Jammer Spectrum:** [`fft_chirp_jammed.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/fft_chirp_jammed.png) (Broadened occupancy covering $43.7\%$ of the monitored band).

---

## 4. In-Phase & Quadrature (IQ) Constellation & Time Series

* **Clean QPSK Constellation:** [`iq_qpsk_clean.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/iq_qpsk_clean.png) (4 well-separated quadrants on unit circle).
* **Noisy BPSK Constellation:** [`iq_bpsk_noisy.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/iq_bpsk_noisy.png) (Antipodal 1D distribution along horizontal axis with Gaussian noise cloud).
* **Clean 16-QAM Constellation:** [`iq_16qam.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/iq_16qam.png) (4x4 rectangular grid showing 3 discrete amplitude concentric circles).
* **Pulsed Burst Interference:** [`iq_burst_interference.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/iq_burst_interference.png) (Radial power spikes reaching $6\times$ nominal amplitude).
* **Swept Chirp Jammed Constellation:** [`iq_chirp_jammed.png`](file:///c:/projects/spectrum%20projects/RF_Spectrum_Monitoring_System/results/plots/iq_chirp_jammed.png) (Continuous spiral phase-plane dispersion).
