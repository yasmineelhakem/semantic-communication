
import os
import csv
import torch
import torch.nn.functional as F

from src.datasets.image_dataset import get_image_dataloaders
from src.models.semantic.semantic_model import SemanticCommunicationModel
from src.evaluation.semantic_classifier import CIFAR10Classifier


SNR_VALUES = [-5, 0, 5, 10, 15, 20]

COMMUNICATION_MODELS = {
    "MSE": "checkpoints/semantic_model.pth",
    "MSE + Perceptual 0.01":
        "checkpoints/semantic_model_perceptual_001.pth"
}

CLASSIFIER_PATH = "checkpoints/semantic_classifier.pth"


def load_classifier(device):

    classifier = CIFAR10Classifier().to(device)

    classifier.load_state_dict(
        torch.load(CLASSIFIER_PATH, map_location=device)
    )

    classifier.eval()

    for parameter in classifier.parameters():
        parameter.requires_grad = False

    return classifier


def load_communication_model(checkpoint, device):

    model = SemanticCommunicationModel().to(device)

    model.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )

    model.eval()

    return model


def evaluate_model(
        model,
        classifier,
        test_loader,
        snr_db,
        device
):

    original_correct = 0
    reconstructed_correct = 0
    preserved = 0
    total = 0

    # Same random channel realization sequence
    # can be reused for the two communication models.
    torch.manual_seed(1234 + int(snr_db))

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Reconstruct images through the AWGN channel
            reconstructed = model(images, snr_db)

            # Classifier was trained on 32x32 CIFAR-10 images
            original_small = F.interpolate(
                images,
                size=(32, 32),
                mode="bilinear",
                align_corners=False
            )

            reconstructed_small = F.interpolate(
                reconstructed,
                size=(32, 32),
                mode="bilinear",
                align_corners=False
            )

            # Predictions on original images
            original_outputs = classifier(original_small)
            original_predictions = original_outputs.argmax(dim=1)

            # Predictions on reconstructed images
            reconstructed_outputs = classifier(reconstructed_small)
            reconstructed_predictions = reconstructed_outputs.argmax(dim=1)

            original_correct += (
                    original_predictions == labels
            ).sum().item()

            reconstructed_correct += (
                    reconstructed_predictions == labels
            ).sum().item()

            preserved += (
                    original_predictions == reconstructed_predictions
            ).sum().item()

            total += labels.size(0)

    return {
        "original_accuracy": 100 * original_correct / total,
        "reconstructed_accuracy": 100 * reconstructed_correct / total,
        "semantic_preservation_accuracy": 100 * preserved / total
    }


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            "Train the semantic classifier first."
        )

    _, test_loader = get_image_dataloaders(batch_size=32)

    classifier = load_classifier(device)

    results = []

    for model_name, checkpoint in COMMUNICATION_MODELS.items():

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)

        model = load_communication_model(checkpoint, device)

        for snr in SNR_VALUES:

            scores = evaluate_model(
                model,
                classifier,
                test_loader,
                snr,
                device
            )

            row = {
                "model": model_name,
                "snr_db": snr,
                **scores
            }

            results.append(row)

            print(
                f"{model_name} | SNR: {snr:>3} dB | "
                f"Original Accuracy: "
                f"{scores['original_accuracy']:.2f}% | "
                f"Reconstructed Accuracy: "
                f"{scores['reconstructed_accuracy']:.2f}% | "
                f"SPA: "
                f"{scores['semantic_preservation_accuracy']:.2f}%"
            )

    os.makedirs("results", exist_ok=True)

    output_path = "results/semantic_preservation.csv"

    with open(output_path, "w", newline="") as file:

        fieldnames = [
            "model",
            "snr_db",
            "original_accuracy",
            "reconstructed_accuracy",
            "semantic_preservation_accuracy"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
