# src/plot_curves.py
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

metrics_path = "outputs/metrics/baseline.csv"
out_dir = "outputs/figures/"
os.makedirs(out_dir, exist_ok=True)

epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []

# read CSV
with open(metrics_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        train_acc.append(float(row["train_acc"]))
        val_loss.append(float(row["val_loss"]))
        val_acc.append(float(row["val_acc"]))

epochs = np.array(epochs)

# loss curve
plt.figure(figsize=(6,4))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(out_dir + "baseline_loss.png", dpi=200)

# accuracy curve
plt.figure(figsize=(6,4))
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(out_dir + "baseline_acc.png", dpi=200)

print("Saved:")
print(out_dir + "baseline_loss.png")
print(out_dir + "baseline_acc.png")
