"""
=============================================================================
Module: train.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Phase: Step 6 - Model Training Engine with Differential Phase Preprocessing
=============================================================================

Pipeline:
---------
1. Generates 6,000 balanced 4-Channel IQ frames (1,500 per class: BPSK, QPSK, 8PSK, 16-QAM):
     - Ch 0: I[n]
     - Ch 1: Q[n]
     - Ch 2: Delta_I[n] = cos(Delta_phi[n])
     - Ch 3: Delta_Q[n] = sin(Delta_phi[n])
2. Random carrier phase theta_0 in [0, 2*pi] and AWGN across [4 dB, 26 dB].
3. Trains 1D-CNN with Cross-Entropy Loss, Adam Optimizer, and Cosine Annealing LR.
4. Evaluates Validation Accuracy, plots training curves & confusion matrix.
5. Saves the optimal model checkpoint to:
       ai_model/weights/best_rf_classifier.pt
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ai_model.dataset_builder import build_synthetic_dataset, LABEL_MAP
from ai_model.model import ModulationCNN1D


def plot_training_history(history: dict, output_png_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss Curve
    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="#0284C7", linewidth=1.8)
    ax_loss.plot(epochs, history["val_loss"], label="Val Loss", color="#EF4444", linewidth=1.8, linestyle="--")
    ax_loss.set_title("Training & Validation Loss (With Diff Phase)", fontweight="bold", fontsize=11)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend()

    # Accuracy Curve
    ax_acc.plot(epochs, history["train_acc"], label="Train Accuracy", color="#0D9488", linewidth=1.8)
    ax_acc.plot(epochs, history["val_acc"], label="Val Accuracy", color="#D97706", linewidth=1.8, linestyle="--")
    ax_acc.set_title("Training & Validation Accuracy (%)", fontweight="bold", fontsize=11)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
    print(f"Saved training history curves: {output_png_path}")
    sys.stdout.flush()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_png_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.figure(figsize=(7, 6))

    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100.0

    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title("AMC Confusion Matrix (%) - Diff Phase Augmented", fontsize=12, fontweight="bold")
    plt.colorbar(label="Accuracy (%)")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10, rotation=25)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]})"
            plt.text(j, i, val_str,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=9,
                     color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel("True Modulation Format", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Modulation Format", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix: {output_png_path}")
    sys.stdout.flush()


def train_ai_model():
    print("=" * 75)
    print("STARTING STEP 6: 1D-CNN AMC WITH DIFFERENTIAL PHASE PREPROCESSING")
    print("=" * 75)
    sys.stdout.flush()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "ai_model", "weights")
    plots_dir = os.path.join(base_dir, "results", "plots")
    model_save_path = os.path.join(weights_dir, "best_rf_classifier.pt")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Dataset Generation with 4 Channels
    print("\n[1/4] Generating 4-Channel Differential Phase Synthetic Dataset (6,000 frames)...")
    sys.stdout.flush()
    X_raw, y_raw = build_synthetic_dataset(num_samples_per_class=1500, seq_len=128, snr_range=(4.0, 26.0))
    total_samples = len(X_raw)
    print(f"      Total IQ Frames: {total_samples} | Shape: {X_raw.shape}")
    print(f"      Channels: Ch0=I, Ch1=Q, Ch2=Delta_I (cos), Ch3=Delta_Q (sin)")
    sys.stdout.flush()

    # 2. Train/Validation Split (80% / 20%)
    split_idx = int(0.8 * total_samples)
    X_train, y_train = X_raw[:split_idx], y_raw[:split_idx]
    X_val, y_val = X_raw[split_idx:], y_raw[split_idx:]
    print(f"      Train Set: {len(X_train)} frames | Validation Set: {len(X_val)} frames")
    sys.stdout.flush()

    # 3. Create PyTorch DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 4. Instantiate 1D-CNN Model (4 input channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/4] Initializing 1D-CNN (input_channels=4) on device: {device}...")
    sys.stdout.flush()
    model = ModulationCNN1D(num_classes=4, input_channels=4, seq_len=128).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 18
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # 5. Training Loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n[3/4] Training 1D-CNN across {num_epochs} Epochs...")
    print(f"      {'Epoch':<8} {'Train Loss':<13} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc'}")
    print("      " + "-" * 55)
    sys.stdout.flush()

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc  = (correct_train / total_train) * 100.0

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc  = (correct_val / total_val) * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        scheduler.step()

        print(f"      #{epoch:<7} {epoch_train_loss:<13.4f} {epoch_val_loss:<12.4f} {epoch_train_acc:<12.1f}% {epoch_val_acc:.1f}%")
        sys.stdout.flush()

        # Save optimal model checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)

    elapsed = time.time() - start_time
    print(f"\n      Training Complete in {elapsed:.1f}s | Optimal Validation Accuracy: {best_val_acc:.1f}%")
    print(f"      Saved Best Checkpoint: {model_save_path}")
    sys.stdout.flush()

    # 6. Confusion Matrix Evaluation
    print("\n[4/4] Evaluating Multi-Class Confusion Matrix on Validation Set...")
    sys.stdout.flush()
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    num_classes = 4
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    hist_png = os.path.join(plots_dir, "ai_training_history.png")
    cm_png   = os.path.join(plots_dir, "ai_confusion_matrix.png")
    plot_training_history(history, hist_png)
    plot_confusion_matrix(cm, [LABEL_MAP[i] for i in range(num_classes)], cm_png)

    print("\n" + "=" * 75)
    print("STEP 6 MODEL TRAINING & EVALUATION COMPLETE!")
    print(f"Validation Accuracy: {best_val_acc:.1f}%")
    print("=" * 75)
    sys.stdout.flush()


if __name__ == "__main__":
    train_ai_model()
