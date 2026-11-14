import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.image_dataset import get_image_dataloaders
from src.models.semantic.semantic_model import SemanticCommunicationModel


def train():

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Hyperparameters
    epochs = 3
    learning_rate = 0.001
    batch_size = 32

    # Load dataset
    train_loader, test_loader = get_image_dataloaders(
        batch_size=batch_size
    )

    # Initialize model
    model = SemanticCommunicationModel().to(device)

    # Reconstruction loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Training
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for images, _ in train_loader:

            images = images.to(device)

            # Random SNR between 0 and 20 dB
            snr_db = random.uniform(0, 20)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed = model(images, snr_db)

            # Calculate loss
            loss = criterion(reconstructed, images)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Training Loss: {average_loss:.6f}"
        )

    # Save model
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        model.state_dict(),
        "checkpoints/semantic_model.pth"
    )

    print("Model saved successfully!")

    # Evaluation
    model.eval()
    test_loss = 0.0

    with torch.no_grad():

        for images, _ in test_loader:

            images = images.to(device)

            reconstructed = model(images, snr_db=10)

            loss = criterion(reconstructed, images)

            test_loss += loss.item()

    average_test_loss = test_loss / len(test_loader)

    print(f"Test Loss at 10 dB: {average_test_loss:.6f}")


if __name__ == "__main__":
    train()
