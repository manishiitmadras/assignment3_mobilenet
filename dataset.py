import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

def get_loaders(batch_size=128, num_workers=4):
    # Stronger augmentation for higher accuracy
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_tf
    )

    return (
        DataLoader(
            train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        ),
        DataLoader(
            test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
    )
