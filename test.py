# src/test.py
import torch
from tqdm import tqdm

from model import build_model
from dataset import get_loaders

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_loader = get_loaders(batch_size=256)

    model = build_model().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/baseline.pt", map_location=device))
    model.eval()

    correct, total = 0, 0
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    print(f"\nFinal Test Top-1 Accuracy: {100 * correct/total:.2f}%")

if __name__ == "__main__":
    main()
