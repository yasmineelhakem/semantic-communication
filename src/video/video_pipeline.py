import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from skimage.metrics import structural_similarity

from src.models.semantic.semantic_model import SemanticCommunicationModel
from src.evaluation.semantic_classifier import CIFAR10Classifier

VIDEO_PATH = "data/videos/input/sample.mp4"

OUTPUT_DIR = "results/videos"

CHECKPOINTS = {
    "MSE": "checkpoints/semantic_model.pth",
    "MSE_Perceptual": "checkpoints/semantic_model_perceptual_001.pth"
}

SNR_VALUES = [-5, 0, 5, 10, 15, 20]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Limite le nombre de frames pour les premières expériences.
# Mettre None pour traiter toute la vidéo.
MAX_FRAMES = 100

FRAME_SIZE = (128, 128)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


def load_classifier():
    classifier = CIFAR10Classifier().to(DEVICE)

    checkpoint = torch.load(
        "checkpoints/semantic_classifier.pth",
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    classifier.load_state_dict(checkpoint)
    classifier.eval()

    return classifier

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la vidéo : {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize to 128x128
        frame = cv2.resize(
            frame,
            FRAME_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # Convert to float [0, 1]
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

        if MAX_FRAMES is not None and len(frames) >= MAX_FRAMES:
            break

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("Aucune frame extraite de la vidéo.")

    print(f"Nombre de frames extraites : {len(frames)}")
    print(f"FPS original : {fps:.2f}")

    return frames, fps

def calculate_psnr(original, reconstructed):

    mse = np.mean(
        (original - reconstructed) ** 2
    )

    if mse == 0:
        return 100.0

    return 10 * np.log10(1.0 / mse)


def calculate_ssim(original, reconstructed):

    return structural_similarity(
        original,
        reconstructed,
        channel_axis=2,
        data_range=1.0
    )

def predict_class(classifier, frame):

    tensor = torch.from_numpy(
        frame
    ).permute(2, 0, 1).unsqueeze(0).float()

    tensor = tensor.to(DEVICE)

    # The classifier was trained on 32x32 CIFAR-10 images.
    tensor = F.interpolate(
        tensor,
        size=(32, 32),
        mode="bilinear",
        align_corners=False
    )

    with torch.no_grad():
        logits = classifier(tensor)
        prediction = logits.argmax(dim=1)

    return prediction.item()

def save_video(frames, output_path, fps):

    height, width, _ = frames[0].shape

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Impossible de créer la vidéo : {output_path}"
        )

    for frame in frames:

        frame_uint8 = np.clip(
            frame * 255.0,
            0,
            255
        ).astype(np.uint8)

        frame_bgr = cv2.cvtColor(
            frame_uint8,
            cv2.COLOR_RGB2BGR
        )

        writer.write(frame_bgr)

    writer.release()

def evaluate_video_model(
        model,
        classifier,
        original_frames,
        fps,
        model_name,
        snr_db
):

    reconstructed_frames = []

    psnr_values = []
    ssim_values = []

    semantic_matches = 0

    original_predictions = [
        predict_class(classifier, frame)
        for frame in original_frames
    ]

    # Evaluate each frame independently.
    with torch.no_grad():

        for index, frame in enumerate(original_frames):

            tensor = torch.from_numpy(
                frame
            ).permute(2, 0, 1).unsqueeze(0).float()

            tensor = tensor.to(DEVICE)

            reconstructed = model(
                tensor,
                snr_db
            )

            reconstructed = reconstructed.clamp(0, 1)

            reconstructed = reconstructed.squeeze(0)
            reconstructed = reconstructed.permute(1, 2, 0)

            reconstructed = reconstructed.cpu().numpy()

            reconstructed_frames.append(reconstructed)

            # Visual quality metrics
            psnr_values.append(
                calculate_psnr(frame, reconstructed)
            )

            ssim_values.append(
                calculate_ssim(frame, reconstructed)
            )

            # Semantic preservation
            reconstructed_prediction = predict_class(
                classifier,
                reconstructed
            )

            if reconstructed_prediction == original_predictions[index]:
                semantic_matches += 1

            if (index + 1) % 10 == 0:
                print(
                    f"Frame {index + 1}/{len(original_frames)}"
                )

    # Average metrics over all frames
    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    spa = (
                  semantic_matches / len(original_frames)
          ) * 100.0

    # Save reconstructed videos
    output_path = os.path.join(
        OUTPUT_DIR,
        model_name,
        f"video_{snr_db}dB.mp4"
    )

    save_video(
        reconstructed_frames,
        output_path,
        fps
    )

    return {
        "model": model_name,
        "snr_db": snr_db,
        "frames": len(original_frames),
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "spa": spa,
        "video_path": output_path
    }

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Using device:", DEVICE)

    # Extract original videos frames once
    original_frames, fps = extract_frames(VIDEO_PATH)

    # Load semantic evaluation classifier
    classifier = load_classifier()

    results = []

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print("\n" + "=" * 55)
        print("MODEL:", model_name)
        print("=" * 55)

        model = SemanticCommunicationModel()

        model = load_checkpoint(
            model,
            checkpoint_path
        )

        for snr_db in SNR_VALUES:

            print(
                f"\nEvaluating {model_name} at SNR = {snr_db} dB"
            )

            # Use reproducible noise realizations.
            # Resetting the seed allows both models to use
            # the same random sequence for corresponding frames.
            torch.manual_seed(1234 + snr_db)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1234 + snr_db)

            result = evaluate_video_model(
                model=model,
                classifier=classifier,
                original_frames=original_frames,
                fps=fps,
                model_name=model_name,
                snr_db=snr_db
            )

            results.append(result)

            print(
                f"PSNR: {result['psnr']:.2f} dB | "
                f"SSIM: {result['ssim']:.4f} | "
                f"SPA: {result['spa']:.2f}%"
            )

    # Save CSV results
    results_df = pd.DataFrame(results)

    csv_path = os.path.join(
        OUTPUT_DIR,
        "video_evaluation.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\n" + "=" * 55)
    print("VIDEO EVALUATION COMPLETED")
    print("=" * 55)

    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
