import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.image_dataset import get_image_dataloaders
from src.models.semantic.semantic_model import SemanticCommunicationModel
from src.models.semantic.losses import PerceptualLoss


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    epochs = 3
    learning_rate = 0.001
    batch_size = 32
    lambda_perceptual = 0.1

    train_loader, test_loader = get_image_dataloaders(
        batch_size=batch_size
    )

    model = SemanticCommunicationModel().to(device)

    # Pixel reconstruction loss
    mse_criterion = nn.MSELoss()

    # Perceptual feature loss
    perceptual_criterion = PerceptualLoss().to(device)
    perceptual_criterion.eval()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_perceptual = 0.0

        for images, _ in train_loader:

            images = images.to(device)

            snr_db = random.uniform(0, 20)

            optimizer.zero_grad()

            reconstructed = model(images, snr_db)

            # Calculate both losses
            mse_loss = mse_criterion(reconstructed, images)

            perceptual_loss = perceptual_criterion(
                images,
                reconstructed
            )

            # Combined objective
            loss = (
                    mse_loss
                    + lambda_perceptual * perceptual_loss
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_perceptual += perceptual_loss.item()

        n = len(train_loader)

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Total Loss: {total_loss / n:.6f}")
        print(f"MSE Loss: {total_mse / n:.6f}")
        print(f"Perceptual Loss: {total_perceptual / n:.6f}")

    # Save the new model separately from the baseline
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        model.state_dict(),
        "checkpoints/semantic_model_perceptual.pth"
    )

    print("\nPerceptual model saved successfully!")

    # Evaluate on test dataset at 10 dB
    model.eval()

    test_mse = 0.0
    test_perceptual = 0.0

    with torch.no_grad():

        for images, _ in test_loader:

            images = images.to(device)

            reconstructed = model(images, snr_db=10)

            test_mse += mse_criterion(
                reconstructed, images
            ).item()

            test_perceptual += perceptual_criterion(
                images, reconstructed
            ).item()

    n_test = len(test_loader)

    print("\nTest results at 10 dB:")
    print(f"MSE: {test_mse / n_test:.6f}")
    print(f"Perceptual Loss: {test_perceptual / n_test:.6f}")


if __name__ == "__main__":
    train()

