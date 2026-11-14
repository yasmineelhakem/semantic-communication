"""Verify preservation, numerical equivalence, and compatibility after relocation."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class ReorganizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = json.loads((ROOT / "docs/original_sources.json").read_text(encoding="utf-8"))
        cls.modules = json.loads((ROOT / "docs/module_map.json").read_text(encoding="utf-8"))

    def test_all_original_artifacts_are_intact(self):
        records = json.loads((ROOT / "docs/reorganization_manifest.json").read_text())
        for record in records:
            old = record["original"]
            if "__pycache__/" in old:
                continue  # Interpreter caches are regenerated when imports run.
            if old.endswith(".py"):
                self.assertIn(old, self.originals)
                if old == "src/channels/awgn.py":
                    # Intentional merge: preserve both AWGN interfaces.
                    from src.channels.awgn import (
                        awgn,
                        awgn_channel,
                        normalize_power,
                        measure_snr_db,
                    )

                    for func in (awgn, awgn_channel, normalize_power, measure_snr_db):
                        self.assertTrue(callable(func))

                elif old not in {f"scripts/{name}.py" for name in self.modules}:
                    self.assertEqual(
                        self.originals[old],
                        (ROOT / old).read_text(encoding="utf-8"),
                    )
                continue
            path = ROOT / record["current"]
            self.assertTrue(path.is_file(), str(path))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], str(path))

    def test_old_imports_resolve_to_new_modules(self):
        for name, target in self.modules.items():
            self.assertIs(importlib.import_module(f"scripts.{name}"), importlib.import_module(target))

    def test_function_and_architecture_bodies_are_preserved(self):
        for name, target in self.modules.items():
            old_tree = ast.parse(self.originals[f"scripts/{name}.py"])
            new_tree = ast.parse((ROOT / (target.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
            new_defs = {n.name: n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for node in old_tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                if node.name in ("ImageDataset", "SyntaxicJSCC"):
                    module = "dataset" if node.name == "ImageDataset" else "model"
                    tree = ast.parse((ROOT / f"src/jscc/{module}.py").read_text(encoding="utf-8"))
                    relocated = next(n for n in tree.body if isinstance(n, ast.ClassDef))
                else:
                    relocated = new_defs[node.name]
                # Only locations and directory creation may differ within main().
                if node.name == "main":
                    continue
                self.assertEqual(ast.dump(node), ast.dump(relocated), f"{target}:{node.name}")

    def test_seeded_model_and_training_step_match_original(self):
        from src.jscc.model import SyntaxicJSCC
        original = {"__name__": "original_training"}
        exec(compile(self.originals["scripts/train_jscc.py"], "original_training", "exec"), original)
        torch.manual_seed(7)
        old = original["SyntaxicJSCC"]()
        new = SyntaxicJSCC()
        new.load_state_dict(old.state_dict(), strict=True)
        images = torch.rand(1, 3, 128, 128)
        optimizers = [torch.optim.Adam(m.parameters(), lr=0.001) for m in (old, new)]
        outputs = []
        for model, optimizer in zip((old, new), optimizers):
            torch.manual_seed(42)
            output = model(images, 10.0)
            loss = torch.nn.functional.mse_loss(output, images)
            loss.backward()
            optimizer.step()
            outputs.append(output.detach())
        self.assertTrue(torch.equal(*outputs))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)

    def test_checkpoint_and_dataset(self):
        from src.jscc.dataset import ImageDataset
        from src.jscc.model import SyntaxicJSCC
        from src.paths import project_path
        model = SyntaxicJSCC()
        checkpoint = torch.load(project_path("results/jscc/best_model.pth"), weights_only=True)
        model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
        dataset = ImageDataset(project_path("data/train_images"))
        self.assertGreater(len(dataset), 0)
        self.assertEqual(tuple(dataset[0].shape), (3, 128, 128))

    def test_classical_single_bit_correction(self):
        from src.classical.hamming74 import encode_hamming74, decode_hamming74
        bits = np.array([int(b) for i in range(16) for b in f"{i:04b}"], dtype=np.uint8)
        encoded, padding = encode_hamming74(bits)
        for position in range(7):
            noisy = encoded.reshape(-1, 7).copy()
            noisy[:, position] ^= 1
            decoded, _ = decode_hamming74(noisy.reshape(-1), padding)
            np.testing.assert_array_equal(decoded, bits)


if __name__ == "__main__":
    unittest.main()
