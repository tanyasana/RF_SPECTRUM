"""
=============================================================================
Module: model.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 6 - Deep Learning 1D-CNN Model Architecture for AMC
=============================================================================

Theoretical Architecture & Mathematical Pipeline with Differential Phase Preprocessing:
--------------------------------------------------------------------------------------
Input Tensor:
    Shape: (Batch_Size, 4, 128)
    - Channel 0: In-phase I[n] time-series samples (RMS normalized)
    - Channel 1: Quadrature Q[n] time-series samples (RMS normalized)
    - Channel 2: Differential In-Phase Delta_I[n] = cos(Delta_phi[n])
    - Channel 3: Differential Quadrature Delta_Q[n] = sin(Delta_phi[n])
    - Sequence Length: 128 consecutive time samples

Why 4 Channels with Differential Phase?
---------------------------------------
Raw IQ representations alone suffer from rotational ambiguity: when carrier phase
rotates arbitrarily (theta_0 in [0, 2*pi]), QPSK (4 points) overlaps with 8PSK.
By appending the normalized conjugate product:
    p[n] = z[n] * conj(z[n-1])
    Delta_I[n] = Re(p[n]) / |p[n]|
    Delta_Q[n] = Im(p[n]) / |p[n]|
the static phase offset theta_0 cancels out identically (theta_0 - theta_0 = 0).
The 1D-CNN retains both amplitude envelope dynamics (Channels 0 & 1) and phase
jump orthogonality (Channels 2 & 3).

Layer Specifications:
  1. Conv1D(4 -> 64, kernel=5, pad=2) + BatchNorm1d + ReLU + MaxPool1d(2)  [Len: 64]
  2. Conv1D(64 -> 128, kernel=3, pad=1) + BatchNorm1d + ReLU + MaxPool1d(2) [Len: 32]
  3. Conv1D(128 -> 128, kernel=3, pad=1) + BatchNorm1d + ReLU + MaxPool1d(2) [Len: 16]
  4. Flatten: 128 channels * 16 samples = 2048 features
  5. Dense(2048 -> 128) + ReLU + Dropout(p=0.4)
  6. Output Dense(128 -> 4 classes)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class ModulationCNN1D(nn.Module):
        def __init__(self, num_classes: int = 4, input_channels: int = 4, seq_len: int = 128):
            super(ModulationCNN1D, self).__init__()
            self.num_classes = num_classes
            self.input_channels = input_channels
            self.seq_len = seq_len

            # Feature Extraction Block 1 (Kernel 5 captures adjacent symbol transitions)
            self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(64)
            self.pool1 = nn.MaxPool1d(kernel_size=2)  # 128 -> 64

            # Feature Extraction Block 2
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(128)
            self.pool2 = nn.MaxPool1d(kernel_size=2)  # 64 -> 32

            # Feature Extraction Block 3
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.MaxPool1d(kernel_size=2)  # 32 -> 16

            # Dense Classification Head
            flattened_dim = 128 * (seq_len // 8)  # 128 * 16 = 2048
            self.fc1 = nn.Linear(flattened_dim, 128)
            self.dropout = nn.Dropout(p=0.4)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            Input x: (Batch, input_channels=4, seq_len=128)
            Returns: Raw logits (Batch, num_classes)
            """
            # Block 1
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            # Block 2
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            # Block 3
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))

            # Flatten
            x = x.view(x.size(0), -1)

            # Fully Connected Head
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            logits = self.fc2(x)
            return logits

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """
            Applies Softmax to output true probability distribution: sum(P) = 1.0
            """
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
            return probs
else:
    class ModulationCNN1D:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to initialize ModulationCNN1D.")
