"""Load existing image models once for independent video-frame inference.

No training, temporal state, or pretrained VGG download is needed here.
The semantic objective is determined by the supplied checkpoint, not a flag
that changes the network during inference.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.metrics.image_quality import calculer_metriques
from src.paths import project_path

MODEL_NAMES = ("classical", "jscc", "semantic-mse", "semantic-perceptual")
DEFAULT_CHECKPOINTS = {
    "jscc": "results/jscc/best_model.pth",
    "semantic-mse": "checkpoints/semantic_model.pth",
    "semantic-perceptual": "checkpoints/semantic_model_perceptual_001.pth",
}


def default_video_output(model):
    if model in ("classical", "jscc"):
        return project_path(f"results/{model}/video")
    return project_path(f"results/semantic/video/{model.removeprefix('semantic-')}")


class NeuralFrameTransmitter:
    """A stateless CPU adapter using the repository's exact checkpoint formats."""

    def __init__(self, model_name, checkpoint=None):
        if model_name not in DEFAULT_CHECKPOINTS:
            raise ValueError(f"Unknown neural model: {model_name}")
        self.checkpoint = (
            Path(checkpoint) if checkpoint is not None
            else project_path(DEFAULT_CHECKPOINTS[model_name])
        ).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {model_name} not found: {self.checkpoint}. "
                "Restore the trained weights or pass --checkpoint PATH. "
                "No untrained substitute will be used."
            )
        # Network construction initializes parameters before loading. Preserve
        # the caller's random state; the video runner seeds channel noise.
        with torch.random.fork_rng(devices=[]):
            if model_name == "jscc":
                from src.jscc.model import SyntaxicJSCC
                self.network = SyntaxicJSCC()
            else:
                from src.models.semantic.semantic_model import SemanticCommunicationModel
                self.network = SemanticCommunicationModel()
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            try:
                if model_name == "jscc":
                    self.network.encoder.load_state_dict(state["encoder"], strict=True)
                    self.network.decoder.load_state_dict(state["decoder"], strict=True)
                else:
                    self.network.load_state_dict(state, strict=True)
            except (KeyError, TypeError, RuntimeError) as exc:
                raise ValueError(
                    f"Checkpoint does not match {model_name}: {self.checkpoint}"
                ) from exc
        self.network.eval()

    @torch.inference_mode()
    def __call__(self, reference, output_dir, snr_db):
        reference, output_dir = Path(reference), Path(output_dir)
        with Image.open(reference) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        reconstructed = self.network(tensor, snr_db)
        if reconstructed.shape != tensor.shape or not torch.isfinite(reconstructed).all():
            raise RuntimeError("Neural model returned invalid pixels or frame dimensions.")
        received = (
            reconstructed[0].permute(1, 2, 0).clamp(0, 1)
            .mul(255).round().to(torch.uint8).cpu().numpy()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / f"{reference.stem}.png"
        Image.fromarray(received).save(png_path)
        psnr, ssim = calculer_metriques(reference, png_path)
        return received, {
            "snr_db": snr_db, "frame_received": True,
            # Neural methods have no JPEG bitstream; these fields are N/A.
            "jpeg_decoded": None, "failure_reason": "",
            "bit_errors": None, "ber": None, "psnr_db": psnr, "ssim": ssim,
        }
