"""Exercise every experiment in temporary folders without replacing saved results."""
import contextlib
import importlib
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from torch.utils.data import Subset

from src.paths import PROJECT_ROOT


def main():
    torch.set_num_threads(2)
    modules = [
        "src.channels.demo_awgn", "src.classical.ber",
        "src.classical.image_transmission", "src.classical.fec_experiment",
        "src.classical.pipeline", "scripts.jscc.test_encoder",
        "scripts.jscc.test_norm", "scripts.jscc.test_awgn_jscc",
        "scripts.jscc.test_decoder", "src.jscc.training",
        "src.jscc.inference", "src.jscc.evaluation", "src.jscc.plotting",
    ]
    with tempfile.TemporaryDirectory(prefix="semantic-communication-check-") as folder:
        temporary = Path(folder)

        def location(relative):
            return (temporary if str(relative).startswith("results/") else PROJECT_ROOT) / relative

        for name in modules:
            module = importlib.import_module(name)
            with contextlib.ExitStack() as stack:
                # Redirect existing output constants; read the preserved checkpoint.
                for key, value in list(vars(module).items()):
                    if isinstance(value, Path) and value.is_relative_to(PROJECT_ROOT / "results"):
                        if key != "MODEL_PATH":
                            stack.enter_context(patch.object(module, key, temporary / value.relative_to(PROJECT_ROOT)))
                if hasattr(module, "project_path"):
                    stack.enter_context(patch.object(module, "project_path", location))
                if name == "src.jscc.training":
                    dataset_type = module.ImageDataset
                    stack.enter_context(patch.object(module, "EPOCHS", 1))
                    stack.enter_context(patch.object(module, "ImageDataset",
                        lambda path: Subset(dataset_type(path), range(5))))
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            print(f"PASS {name}")
        expected = [
            "results/classical/image/reconstruction.png",
            "results/classical/fec/metriques.csv",
            "results/jscc/best_model.pth",
            "results/jscc/history.csv",
            "results/jscc/evaluation/metrics.csv",
            "results/jscc/evaluation_multi/summary.csv",
            "results/jscc/graphs/graph1/training_loss.png",
            "results/jscc/graphs/graph2/psnr_multi.png",
        ]
        for path in expected:
            assert (temporary / path).is_file(), path
        print("PASS expected output files; temporary outputs removed on exit")


if __name__ == "__main__":
    main()
