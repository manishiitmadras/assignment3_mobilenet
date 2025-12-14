# MobileNet Training Pipeline

A PyTorch training script for MobileNet with reproducibility, checkpointing, and learning rate scheduling.

---

## Features

- MobileNet model (`build_model()` in `model.py`)  
- Dataset loaders (`get_loaders()` in `dataset.py`)  
- Fixed seed (42) for reproducibility  
- Cross-entropy loss with label smoothing (0.1)  
- SGD optimizer with momentum, weight decay, and Nesterov acceleration  
- Gradient clipping (5.0) for stable training  
- Learning rate scheduling: Linear warmup + Cosine annealing  
- Saves best model checkpoint based on validation accuracy  
- Logs metrics (loss, accuracy, learning rate) to CSV  

---

## Environment & Installation

**Recommended environment:** Python 3.10+, PyTorch 2.x, GPU for faster training  

Install dependencies using pip:

```bash
pip install torch torchvision numpy tqdm
