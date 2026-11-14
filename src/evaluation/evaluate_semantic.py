import os
import csv
import torch

from src.datasets.image_dataset import get_image_dataloaders
from src.models.semantic.semantic_model import SemanticCommunicationModel
from src.models.semantic.losses import PerceptualLoss

from src.evaluation.metrics import (
    calculate_psnr,
    calculate_ssim
)


def load_model(checkpoint_path, device):

    model = SemanticCommunicationModel().to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    model.eval()

    return model


def evaluate_model(
        model,
        test_loader,
        snr_values,
        perceptual_criterion,
        device,
        model_name
):

    results = []

    for snr_db in snr_values:

        # Ensure reproducible and comparable channel noise
        seed = 1234 + (snr_db + 5) * 100

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        total_psnr = 0.0
        total_ssim = 0.0
        total_perceptual = 0.0

        print(
            f"{model_name} | Evaluating SNR = {snr_db} dB"
        )

        with torch.no_grad():

            for images, _ in test_loader:

                images = images.to(device)

                reconstructed = model(images, snr_db)

                batch_size_actual = images.size(0)

                # Calculate PSNR
                total_psnr += calculate_psnr(
                    images, reconstructed
                ) * batch_size_actual

                # Calculate SSIM
                total_ssim += calculate_ssim(
                    images, reconstructed
                ) * batch_size_actual

                # Calculate Perceptual Loss
                total_perceptual += perceptual_criterion(
                    images, reconstructed
                ).item() * batch_size_actual

        number_of_images = len(test_loader.dataset)

        results.append({
            "model": model_name,
            "snr_db": snr_db,
            "psnr": total_psnr / number_of_images,
            "ssim": total_ssim / number_of_images,
            "perceptual_loss": (
                    total_perceptual / number_of_images
            )
        })

    return results


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load test dataset
    _, test_loader = get_image_dataloaders(
        batch_size=32
    )

    # Load baseline model (MSE only)
    baseline = load_model(
        "checkpoints/semantic_model.pth",
        device
    )

    # Load perceptual model (MSE + Perceptual Loss)
    perceptual_model = load_model(
        "checkpoints/semantic_model_perceptual_001.pth",
        device
    )

    # Initialize perceptual loss
    perceptual_criterion = PerceptualLoss().to(device)
    perceptual_criterion.eval()

    # SNR values to evaluate
    snr_values = [-5, 0, 5, 10, 15, 20]

    # Evaluate baseline model
    baseline_results = evaluate_model(
        baseline,
        test_loader,
        snr_values,
        perceptual_criterion,
        device,
        "MSE"
    )

    # Evaluate perceptual model
    perceptual_results = evaluate_model(
        perceptual_model,
        test_loader,
        snr_values,
        perceptual_criterion,
        device,
        "MSE + Perceptual"
    )

    # Combine results
    all_results = baseline_results + perceptual_results

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Save results to CSV
    with open(
            "results/semantic_evaluation.csv",
            "w",
            newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "snr_db",
                "psnr",
                "ssim",
                "perceptual_loss"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Display results
    print("\nEvaluation completed!")
    print("\nResults:")

    for result in all_results:

        print(
            f"{result['model']} | "
            f"SNR: {result['snr_db']:>3} dB | "
            f"PSNR: {result['psnr']:.4f} | "
            f"SSIM: {result['ssim']:.4f} | "
            f"Perceptual: {result['perceptual_loss']:.4f}"
        )

    print(
        "\nResults saved in results/semantic_evaluation.csv"
    )


if __name__ == "__main__":
    main()
