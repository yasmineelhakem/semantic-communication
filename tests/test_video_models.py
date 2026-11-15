"""Check checkpoint selection, stateless inference, and neural video routing."""
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.classical.video_transmission import main, transmit_video
from src.models.semantic.semantic_model import SemanticCommunicationModel
from src.paths import project_path
from src.video.models import NeuralFrameTransmitter, default_video_output

try:
    import cv2
except ImportError:
    cv2 = None


class VideoModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "frame_000000.png"
        self.other = self.root / "frame_000001.png"
        Image.new("RGB", (128, 128), (180, 50, 70)).save(self.reference)
        Image.new("RGB", (128, 128), (20, 180, 50)).save(self.other)

    def checkpoint(self, bias):
        """Synthetic known-output weights test loading, not trained quality."""
        with torch.random.fork_rng(devices=[]):
            network = SemanticCommunicationModel()
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.zero_()
                network.decoder.decoder[4].bias.fill_(bias)
            path = self.root / f"semantic_{bias}.pth"
            torch.save(network.state_dict(), path)
        return path

    def test_semantic_selection_uses_checkpoint_weights(self):
        for name, bias in [("semantic-mse", -2.0), ("semantic-perceptual", 2.0)]:
            with self.subTest(model=name):
                path = self.checkpoint(bias)
                adapter = NeuralFrameTransmitter(name, path)
                pixels, metrics = adapter(self.reference, self.root / name, 10)
                expected = round(torch.sigmoid(torch.tensor(bias)).item() * 255)
                self.assertTrue(np.all(pixels == expected))
                self.assertTrue(metrics["frame_received"])
                self.assertIsNone(metrics["jpeg_decoded"])
                self.assertIsNone(metrics["ber"])
                self.assertFalse(adapter.network.training)
                self.assertEqual(adapter.checkpoint, path.resolve())

    def test_actual_jscc_checkpoint_is_stateless_and_rng_is_preserved_on_load(self):
        checkpoint = project_path("results/jscc/best_model.pth")
        if not checkpoint.is_file():
            self.skipTest("Archived JSCC checkpoint not available")
        state = torch.get_rng_state().clone()
        adapter = NeuralFrameTransmitter("jscc")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33)
            first, metrics = adapter(self.reference, self.root / "first", 10)
            adapter(self.other, self.root / "other", 0)
            torch.manual_seed(33)
            replayed, _ = adapter(self.reference, self.root / "replayed", 10)
        np.testing.assert_array_equal(first, replayed)
        self.assertEqual(first.shape, (128, 128, 3))
        self.assertTrue(np.isfinite(metrics["psnr_db"]))
        self.assertTrue(np.isfinite(metrics["ssim"]))

    def test_missing_or_wrong_weights_fail_without_using_random_model(self):
        with self.assertRaisesRegex(FileNotFoundError, "--checkpoint"):
            NeuralFrameTransmitter("semantic-perceptual", self.root / "absent.pth")
        wrong = self.root / "wrong.pth"
        torch.save({"unrelated": torch.zeros(1)}, wrong)
        for name in ["jscc", "semantic-mse"]:
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "does not match"):
                NeuralFrameTransmitter(name, wrong)

    def test_model_output_directories_are_separate(self):
        self.assertEqual(default_video_output("classical"), project_path("results/classical/video"))
        self.assertEqual(default_video_output("jscc"), project_path("results/jscc/video"))
        self.assertEqual(default_video_output("semantic-mse"), project_path("results/semantic/video/mse"))
        self.assertEqual(default_video_output("semantic-perceptual"),
                         project_path("results/semantic/video/perceptual"))

    @unittest.skipIf(cv2 is None, "Install opencv-python-headless for video routing test")
    def test_cli_selects_neural_model_once_and_preserves_fps(self):
        source = self.root / "input.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                                 12.5, (32, 32))
        if not writer.isOpened():
            writer.release()
            self.skipTest("MP4 codec unavailable")
        try:
            writer.write(np.full((32, 32, 3), 60, dtype=np.uint8))
            writer.write(np.full((32, 32, 3), 150, dtype=np.uint8))
        finally:
            writer.release()
        checkpoint = self.checkpoint(2.0)
        output = self.root / "outputs"
        with patch("src.classical.video_transmission.NeuralFrameTransmitter",
                   wraps=NeuralFrameTransmitter) as loader, patch("sys.stdout", new=StringIO()):
            main(["--input", str(source), "--model", "semantic-perceptual",
                  "--checkpoint", str(checkpoint), "--output", str(output),
                  "--snr", "0", "20"])
        loader.assert_called_once_with("semantic-perceptual", checkpoint)
        run, = output.iterdir()
        metadata = json.loads((run / "run.json").read_text())
        self.assertEqual(metadata["model"], "semantic-perceptual")
        self.assertEqual(metadata["checkpoint"], str(checkpoint.resolve()))
        self.assertIsNone(metadata["fec"])
        self.assertIsNone(metadata["jpeg_quality"])
        with (run / "frames.csv").open(newline="") as stream:
            frames = list(csv.DictReader(stream))
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(row["model"] == "semantic-perceptual" for row in frames))
        self.assertTrue(all(row["ber"] == "" and row["jpeg_decoded"] == "" for row in frames))
        for snr in [0, 20]:
            video = cv2.VideoCapture(str(run / f"reconstruction_{snr}dB.avi"))
            try:
                self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 12.5, places=2)
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                success, pixels = video.read()
                self.assertTrue(success)
                self.assertEqual(pixels.shape, (128, 128, 3))
            finally:
                video.release()

    def test_incompatible_options_are_rejected_before_output_creation(self):
        source = self.root / "exists.mp4"
        source.touch()
        output = self.root / "outputs"
        for kwargs in [
            {"model": "unknown"}, {"model": "jscc", "image_size": 32},
            {"model": "semantic-mse", "fec": False},
            {"model": "semantic-mse", "jpeg_quality": 90},
            {"model": "classical", "checkpoint": "unused.pth"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                transmit_video(source, output, **kwargs)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
