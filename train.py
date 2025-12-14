import torch, random, numpy as np
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import os, csv

from model import build_model
from dataset import get_loaders


# ----------------------------
# FIXED SEED (important!)
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ----------------------------
# TRAIN/EVAL FUNCTIONS
# ----------------------------
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        # Gradient clipping (helps MobileNet!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        opt.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LOADERS
    train_loader, test_loader = get_loaders(batch_size=128)

    # MODEL
    model = build_model().to(device)

    # Improved cross-entropy with label smoothing
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Better optimizer settings for MobileNet
    opt = SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    # Warmup + Cosine scheduler
    warmup_scheduler = LinearLR(opt, start_factor=0.1, total_iters=5)
    main_scheduler = CosineAnnealingLR(opt, T_max=95)

    # OUTPUT FOLDERS
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    metrics_file = "outputs/metrics/baseline.csv"

    # Write header
    with open(metrics_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])


    best_acc = 0

    for epoch in range(100):
        print(f"\nEpoch {epoch} -----------------------")

        train_loss, train_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)

        print(f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        # Save metrics
        with open(metrics_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, opt.param_groups[0]['lr']])

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "outputs/checkpoints/baseline.pt")
            print("Checkpoint saved!")

        # Step schedulers
        if epoch < 5:
            warmup_scheduler.step()
        else:
            main_scheduler.step()


if __name__ == "__main__":
    main()
