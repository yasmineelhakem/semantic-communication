"""Small codec-backed tests; all inputs and outputs live in temporary folders."""
import csv
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.classical.video_transmission import DEFAULT_SNRS, transmit_frame, transmit_video
from src.metrics.image_quality import calculer_metriques

try:
    import cv2
except ImportError:
    cv2 = None


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class FrameTransmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "frame_000000.png"
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:] = [180, 70, 30]
        pixels[8:24, 8:24] = [30, 150, 220]
        Image.fromarray(pixels).save(self.reference)

    def test_existing_hamming_corrects_one_error_per_codeword(self):
        def damage_codewords(bits, snr):
            received = bits.copy()
            received[::7] ^= 1
            return received

        with patch("src.classical.video_transmission.transmettre_qpsk",
                   side_effect=damage_codewords) as channel:
            pixels, metrics = transmit_frame(self.reference, self.root / "received", 10)
        channel.assert_called_once()
        self.assertTrue(metrics["jpeg_decoded"])
        self.assertEqual(metrics["bit_errors"], 0)
        with Image.open(self.reference) as image:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=75)
        self.assertEqual((self.root / "received/frame_000000.jpg").read_bytes(),
                         buffer.getvalue())
        with Image.open(BytesIO(buffer.getvalue())) as image:
            np.testing.assert_array_equal(pixels, np.asarray(image.convert("RGB")))

    def test_failed_jpeg_becomes_black_and_has_real_metrics(self):
        with patch("src.classical.video_transmission.transmettre_qpsk",
                   side_effect=lambda bits, snr: np.zeros_like(bits)):
            pixels, metrics = transmit_frame(self.reference, self.root / "failed", 0)
        self.assertFalse(metrics["jpeg_decoded"])
        self.assertEqual(metrics["failure_reason"], "jpeg_decode_failed")
        self.assertTrue(np.all(pixels == 0))
        expected = calculer_metriques(self.reference, self.root / "failed/frame_000000.png")
        self.assertEqual((metrics["psnr_db"], metrics["ssim"]), expected)

    def test_no_fec_transmits_jpeg_bits_directly(self):
        with patch("src.classical.video_transmission.transmettre_qpsk",
                   side_effect=lambda bits, snr: bits.copy()) as channel:
            _, metrics = transmit_frame(self.reference, self.root / "uncoded", 20, fec=False)
        sent_bits = channel.call_args.args[0]
        expected_bytes = (self.root / "uncoded/frame_000000.jpg").read_bytes()
        self.assertEqual(sent_bits.size, 8 * len(expected_bytes))
        self.assertEqual(metrics["bit_errors"], 0)
        self.assertTrue(metrics["jpeg_decoded"])


@unittest.skipIf(cv2 is None, "Install opencv-python-headless for codec integration tests")
class VideoTransmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "short.mp4"
        writer = cv2.VideoWriter(str(self.input), cv2.VideoWriter_fourcc(*"mp4v"),
                                 29.97, (32, 32))
        if not writer.isOpened():
            writer.release()
            self.skipTest("Input MP4 codec is unavailable on this system")
        try:
            for index in range(3):
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                frame[:] = [30 + index * 25, 80, 180 - index * 20]  # BGR
                frame[8:24, 8:24] = [200, 150, 20]
                writer.write(frame)
        finally:
            writer.release()

    def test_mp4_all_snrs_preserves_fps_count_and_summary(self):
        output = transmit_video(self.input, self.root / "results", image_size=32)
        metadata = json.loads((output / "run.json").read_text())
        self.assertAlmostEqual(metadata["fps"], 29.97, places=2)
        self.assertEqual(metadata["n_frames"], 3)
        self.assertEqual(metadata["snr_db"], list(DEFAULT_SNRS))
        details = read_csv(output / "frames.csv")
        summary = read_csv(output / "summary.csv")
        self.assertEqual(len(details), 15)
        self.assertEqual(len(summary), 5)

        source = cv2.VideoCapture(str(self.input))
        try:
            success, original = source.read()
            self.assertTrue(success)
            expected_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            with Image.open(output / "reference_frames/frame_000000.png") as image:
                np.testing.assert_array_equal(np.asarray(image), expected_rgb)
        finally:
            source.release()

        for row in summary:
            snr = float(row["snr_db"])
            measurements = [r for r in details if float(r["snr_db"]) == snr]
            self.assertEqual([int(r["frame_index"]) for r in measurements], [0, 1, 2])
            self.assertAlmostEqual(float(row["psnr_mean"]),
                                   np.mean([float(r["psnr_db"]) for r in measurements]))
            self.assertAlmostEqual(float(row["ssim_mean"]),
                                   np.mean([float(r["ssim"]) for r in measurements]))
            decoded = sum(r["jpeg_decoded"] == "True" for r in measurements)
            self.assertEqual(int(row["decoded_frames"]), decoded)
            self.assertEqual(int(row["failed_frames"]), 3 - decoded)
            for measurement in measurements:
                index = int(measurement["frame_index"])
                self.assertAlmostEqual(float(measurement["time_seconds"]),
                                       index / metadata["fps"])
                reference = output / "reference_frames" / f"frame_{index:06d}.png"
                received = output / f"snr_{snr:g}dB" / f"frame_{index:06d}.png"
                expected_psnr, expected_ssim = calculer_metriques(reference, received)
                self.assertEqual(float(measurement["psnr_db"]), expected_psnr)
                self.assertEqual(float(measurement["ssim"]), expected_ssim)
                if measurement["jpeg_decoded"] == "False":
                    with Image.open(received) as image:
                        self.assertTrue(np.all(np.asarray(image) == 0))
            video = cv2.VideoCapture(str(output / f"reconstruction_{snr:g}dB.avi"))
            try:
                self.assertTrue(video.isOpened())
                self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), metadata["fps"], places=2)
                count = 0
                while True:
                    success, frame = video.read()
                    if not success:
                        break
                    self.assertEqual(frame.shape, (32, 32, 3))
                    count += 1
                self.assertEqual(count, 3)
            finally:
                video.release()

    def test_failure_after_good_frame_never_reuses_previous_frame(self):
        calls = 0

        def channel(bits, snr):
            nonlocal calls
            calls += 1
            return bits.copy() if calls == 1 else np.zeros_like(bits)

        with patch("src.classical.video_transmission.transmettre_qpsk", side_effect=channel):
            output = transmit_video(self.input, self.root / "results", snr_values=[20],
                                    image_size=32, max_frames=2)
        details = read_csv(output / "frames.csv")
        self.assertEqual([r["jpeg_decoded"] for r in details], ["True", "False"])
        with Image.open(output / "snr_20dB/frame_000001.png") as image:
            self.assertTrue(np.all(np.asarray(image) == 0))
        self.assertEqual(json.loads((output / "run.json").read_text())["n_frames"], 2)

    def test_invalid_inputs_fail_before_creating_results(self):
        output = self.root / "results"
        with self.assertRaises(FileNotFoundError):
            transmit_video(self.root / "missing.mp4", output)
        for kwargs in [{"snr_values": []}, {"snr_values": [float("nan")]},
                       {"snr_values": [5, 5]}, {"image_size": 31},
                       {"max_frames": 0}, {"jpeg_quality": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                transmit_video(self.input, output, **kwargs)
        invalid = self.root / "invalid.mp4"
        invalid.write_bytes(b"not a video")
        with self.assertRaises(ValueError):
            transmit_video(invalid, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
