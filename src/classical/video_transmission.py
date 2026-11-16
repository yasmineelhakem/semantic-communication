"""Independent classical or neural transmission of video frames.

Failed JPEGs become black frames, never copies of a previous frame. Metrics
include these failures and are measured on PNG frames before MJPEG export.
"""
import argparse
from contextlib import ExitStack
import csv
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from src.classical.hamming74 import encode_hamming74, decode_hamming74
from src.classical.pipeline import transmettre_qpsk, sauver_jpeg
from src.metrics.image_quality import calculer_metriques
from src.paths import project_path
from src.video.models import MODEL_NAMES, NeuralFrameTransmitter, default_video_output

DEFAULT_SNRS = (0, 5, 10, 15, 20)
FRAME_COLUMNS = (
    "frame_index", "time_seconds", "model", "snr_db", "frame_received", "jpeg_decoded",
    "failure_reason", "bit_errors", "ber", "psnr_db", "ssim",
)


def transmit_frame(reference, output_dir, snr_db, *, jpeg_quality=75, fec=True):
    """Transmit a single RGB reference PNG; return received RGB pixels and metrics.

    No frame history is accepted or used. The reference must already have the
    desired dimensions. Received JPEG bytes and displayed PNG are retained.
    """
    reference, output_dir = Path(reference), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(reference) as original:
        image = original.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    bits = np.unpackbits(np.frombuffer(buffer.getvalue(), dtype=np.uint8)).copy()
    if fec:
        coded, padding = encode_hamming74(bits)
        received, _ = decode_hamming74(transmettre_qpsk(coded, snr_db), padding)
    else:
        received = transmettre_qpsk(bits, snr_db)

    jpeg_path = output_dir / f"{reference.stem}.jpg"
    png_path = output_dir / f"{reference.stem}.png"
    try:
        decoded = sauver_jpeg(received, jpeg_path, png_path)
    except Image.DecompressionBombError:
        decoded = False
    reason = "" if decoded else "jpeg_decode_failed"
    if decoded:
        with Image.open(png_path) as recovered:
            if recovered.size != image.size:
                decoded, reason = False, "invalid_frame_dimensions"
    if not decoded:
        Image.new("RGB", image.size, color=0).save(png_path)

    psnr, ssim = calculer_metriques(reference, png_path)
    errors = int(np.count_nonzero(bits != received))
    with Image.open(png_path) as recovered:
        pixels = np.array(recovered.convert("RGB"))
    return pixels, {
        "snr_db": snr_db, "frame_received": decoded, "jpeg_decoded": decoded,
        "failure_reason": reason, "bit_errors": errors,
        "ber": errors / bits.size, "psnr_db": psnr, "ssim": ssim,
    }


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video support needs OpenCV: python -m pip install opencv-python-headless"
        ) from exc
    return cv2


def transmit_video(input_path, output_root=None, *, snr_values=DEFAULT_SNRS,
                   image_size=128, jpeg_quality=75, fec=True, seed=42,
                   max_frames=None, model="classical", checkpoint=None):
    """Stream a video once and independently transmit each frame at every SNR.

    Return a fresh run directory under the selected model's video results.
    Only decoded input frames are evaluated. Output is silent MJPEG AVI at the
    source's reported FPS (constant-frame-rate playback), with no inter coding.
    """
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    snrs = tuple(float(value) for value in snr_values)
    if not snrs or len(set(snrs)) != len(snrs) or not all(map(math.isfinite, snrs)):
        raise ValueError("SNR values must be finite, nonempty, and unique.")
    if image_size < 8 or image_size % 2:
        raise ValueError("Frame size must be even and at least 8 (SSIM/MJPEG).")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("JPEG quality must be between 1 and 95.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive.")
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}. Choose from {MODEL_NAMES}.")
    if model == "classical" and checkpoint is not None:
        raise ValueError("Classical transmission does not use a checkpoint.")
    if model != "classical":
        if image_size != 128:
            raise ValueError("Existing neural checkpoints require --size 128.")
        if not fec or jpeg_quality != 75:
            raise ValueError("--no-fec and --jpeg-quality apply only to the classical model.")
    neural = NeuralFrameTransmitter(model, checkpoint) if model != "classical" else None

    cv2 = _opencv()
    with ExitStack() as stack:
        capture = cv2.VideoCapture(str(input_path))
        stack.callback(capture.release)
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input video has no valid FPS; cannot preserve its timing.")
        success, frame = capture.read()
        if not success:
            raise ValueError("Input video contains no decodable frames.")
        source_height, source_width = frame.shape[:2]
        output_root = Path(output_root) if output_root is not None else default_video_output(model)
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=output_root))
        references = run_dir / "reference_frames"
        references.mkdir()
        writers, destinations = {}, {}
        totals = {snr: {"decoded": 0, "psnr": 0.0, "ssim": 0.0} for snr in snrs}
        for snr in snrs:
            tag = f"snr_{snr:g}dB"
            destination = run_dir / tag
            destination.mkdir()
            destinations[snr] = destination
            writer = cv2.VideoWriter(
                str(run_dir / f"reconstruction_{snr:g}dB.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"), fps, (image_size, image_size),
            )
            stack.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError("Cannot initialize MJPEG AVI writer on this system.")
            writers[snr] = writer

        metrics_file = stack.enter_context(
            (run_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        )
        csv_writer = csv.DictWriter(metrics_file, fieldnames=FRAME_COLUMNS)
        csv_writer.writeheader()
        frame_count = 0
        stack.enter_context(torch.random.fork_rng(devices=[]))
        stack.enter_context(torch.no_grad())
        torch.manual_seed(seed)
        while success and (max_frames is None or frame_count < max_frames):
            # OpenCV decodes BGR; the existing image pipeline expects RGB.
            reference = references / f"frame_{frame_count:06d}.png"
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).resize((image_size, image_size)).save(reference)
            for snr in snrs:
                if neural is None:
                    pixels, metrics = transmit_frame(
                        reference, destinations[snr], snr,
                        jpeg_quality=jpeg_quality, fec=fec,
                    )
                else:
                    pixels, metrics = neural(reference, destinations[snr], snr)
                writers[snr].write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                csv_writer.writerow({
                    "frame_index": frame_count, "time_seconds": frame_count / fps,
                    "model": model,
                    **metrics,
                })
                totals[snr]["decoded"] += int(metrics["frame_received"])
                totals[snr]["psnr"] += metrics["psnr_db"]
                totals[snr]["ssim"] += metrics["ssim"]
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
            success, frame = capture.read()

        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            summary = csv.DictWriter(stream, fieldnames=[
                "model", "snr_db", "n_frames", "decoded_frames", "failed_frames",
                "decode_rate", "psnr_mean", "ssim_mean",
            ])
            summary.writeheader()
            for snr in snrs:
                total = totals[snr]
                summary.writerow({
                    "model": model, "snr_db": snr, "n_frames": frame_count,
                    "decoded_frames": total["decoded"],
                    "failed_frames": frame_count - total["decoded"],
                    "decode_rate": total["decoded"] / frame_count,
                    "psnr_mean": total["psnr"] / frame_count,
                    "ssim_mean": total["ssim"] / frame_count,
                })
        metadata = {
            "input": str(input_path), "fps": fps, "n_frames": frame_count,
            "source_width": source_width, "source_height": source_height,
            "model": model, "checkpoint": str(neural.checkpoint) if neural else None,
            "device": "cpu", "frame_size": image_size, "snr_db": snrs,
            "jpeg_quality": jpeg_quality if neural is None else None,
            "fec": ("Hamming (7,4)" if fec else "none") if neural is None else None,
            "seed": seed,
            "max_frames": max_frames, "video_codec": "MJPG", "audio": False,
            "failed_frame_policy": (
                "black frame, included in all-frame averages" if neural is None
                else "invalid network output aborts the run"
            ),
            "metrics_stage": "received PNG before additional MJPEG video encoding",
            "timing": "constant frame rate using source-reported FPS",
        }
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Short MP4 input")
    parser.add_argument("--model", choices=MODEL_NAMES, default="classical",
                        help="Image transmission method (default: classical)")
    parser.add_argument("--checkpoint", type=Path, help="Override the selected neural model's weights")
    parser.add_argument("--output", type=Path, help="Override the model-specific results directory")
    parser.add_argument("--snr", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--size", type=int, default=128, help="Square frame size (default 128)")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--no-fec", action="store_true", help="Disable Hamming (7,4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, help="Process only the first N frames")
    args = parser.parse_args(argv)
    try:
        result = transmit_video(
            args.input, args.output, snr_values=args.snr, image_size=args.size,
            jpeg_quality=args.jpeg_quality, fec=not args.no_fec,
            seed=args.seed, max_frames=args.max_frames,
            model=args.model, checkpoint=args.checkpoint,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Video transmission failed: {exc}\n")
    print(f"Reconstructed videos, per-frame metrics, and averages saved in: {result}")


if __name__ == "__main__":
    main()
