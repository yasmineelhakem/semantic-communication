from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_image_dataloaders(batch_size=32):

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Download and load CIFAR-10 training data
    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    # Download and load CIFAR-10 test data
    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Small subsets for initial experiments
    train_dataset = Subset(train_dataset, range(5000))
    test_dataset = Subset(test_dataset, range(1000))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader
