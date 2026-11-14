import os
import torch
import matplotlib.pyplot as plt

from src.datasets.image_dataset import get_image_dataloaders
from src.models.semantic.semantic_model import SemanticCommunicationModel


def load_model(checkpoint_path, device):
    model = SemanticCommunicationModel().to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    model.eval()
    return model


def set_seed(seed):
    torch.manual_seed(seed)


def display_image(image):
    image = image.detach().cpu().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load test dataset
    _, test_loader = get_image_dataloaders(batch_size=32)

    # Load both trained models
    mse_model = load_model(
        "checkpoints/semantic_model.pth",
        device
    )

    perceptual_model = load_model(
        "checkpoints/semantic_model_perceptual_001.pth",
        device
    )

    # Select one test image
    images, labels = next(iter(test_loader))
    original = images[0:1].to(device)

    snr_values = [-5, 0, 10, 20]

    # Create comparison figure
    fig, axes = plt.subplots(
        len(snr_values), 3,
        figsize=(12, 14)
    )

    for row, snr_db in enumerate(snr_values):

        # Original image
        axes[row, 0].imshow(display_image(original[0]))
        axes[row, 0].set_title("Original")

        # Same AWGN realization for both models
        set_seed(1234 + (snr_db + 5) * 100)
        with torch.no_grad():
            mse_reconstruction = mse_model(original, snr_db)

        set_seed(1234 + (snr_db + 5) * 100)
        with torch.no_grad():
            perceptual_reconstruction = perceptual_model(
                original, snr_db
            )

        axes[row, 1].imshow(
            display_image(mse_reconstruction[0])
        )
        axes[row, 1].set_title("MSE")

        axes[row, 2].imshow(
            display_image(perceptual_reconstruction[0])
        )
        axes[row, 2].set_title("MSE + Perceptual")

        for col in range(3):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"SNR = {snr_db} dB",
            fontsize=12
        )

    plt.suptitle(
        "Semantic Image Communication - Reconstruction Comparison",
        fontsize=14
    )

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    plt.savefig(
        "results/reconstruction_comparison_001.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "Figure saved: results/reconstruction_comparison_001.png"
    )


if __name__ == "__main__":
    main()
