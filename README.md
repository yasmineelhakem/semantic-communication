# Semantic communication: classical transmission and neural JSCC

The project contains two independent image transmission pipelines sharing the same differentiable complex AWGN channel.

- **Classical:** RGB image -> JPEG -> bits -> optional Hamming (7,4) -> Gray QPSK -> AWGN -> nearest-point demodulation -> optional Hamming decoding -> JPEG reconstruction, BER, PSNR and SSIM.
- **Neural JSCC:** RGB 128x128 -> CNN encoder -> complex symbols and power normalization -> AWGN -> CNN decoder -> reconstruction, PSNR and SSIM. Training uses the original MSE loss, Adam optimizer, CPU device, seeds and hyperparameters.

## Project structure

```text
src/
  classical/
    hamming74.py           # Hamming encoder/decoder
    pipeline.py            # Full JPEG/FEC/QPSK experiment with image metrics
    fec_experiment.py      # Original FEC/BER experiment
    image_transmission.py  # Original JPEG experiment without FEC
    ber.py                 # Original QPSK BER experiment
  jscc/
    encoder.py             # Unchanged EncoderCNN architecture
    decoder.py             # Unchanged DecoderCNN architecture
    normalization.py       # Unchanged complex symbol normalization
    model.py               # SyntaxicJSCC end-to-end model
    dataset.py             # Original ImageDataset and preprocessing
    training.py            # Training, validation and checkpoint saving
    inference.py           # Checkpoint inference and single-image evaluation
    evaluation.py          # Multi-image/repeated-noise evaluation and statistics
    plotting.py            # Training/evaluation figures
  channels/
    awgn.py                # Shared complex AWGN, power and SNR functions
    demo_awgn.py           # QPSK constellation demonstration
  metrics/image_quality.py # Image PSNR/SSIM utility
  paths.py                 # Locations resolved relative to the project root
scripts/
  classical/               # Classical experiment entry points
  jscc/                    # JSCC entry points and original component diagnostics
  shared/                  # Shared channel demonstration
  *.py                     # Compatibility entry points for all original commands
  verify_reorganization.py # Safe smoke checks using temporary output folders
data/                      # Preserved in its original location
  input.jpg
  train_images/
  test_images/
  test_images_multi/
results/                   # All existing artifacts retained, grouped by pipeline
  classical/
    constellations/         # Original constellation_*dB.png files
    image/                  # Original JPEGs, reference and reconstruction
    fec/                    # Original FEC images, received JPEGs and metriques.csv
  jscc/
    best_model.pth          # Original trained checkpoint, unchanged
    history.csv            # Original training history, unchanged
    evaluation/            # Single-image metrics and reconstructions
    evaluation_multi/      # Detailed measurements and summary statistics
    graphs/
      graph1/              # Original single-image and training figures
      graph2/              # Original multi-image figures
tests/                     # AWGN and preservation/regression checks
docs/
   reorganization_manifest.json # Original/current artifact paths, sizes and SHA-256
   module_map.json               # Original script to implementation mapping
   original_sources.json        # Complete original Python source archive
   README.original.md           # Original README
reorganize_project.py       # Guarded one-time migration record; do not rerun
```

## Setup

Run commands from the project root (the directory containing this README).

Windows PowerShell:

```powershell
# Reuse the existing .venv, or create it if needed:
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are PyTorch, matplotlib, NumPy, Pillow and scikit-image. The reorganization was verified with the existing Python 3.14 / PyTorch 2.15 CPU environment.

## Classical experiments

Place an image at `data/input.jpg` (the existing image is retained).

To test the full classical pipeline (JPEG + optional Hamming (7,4) + QPSK + AWGN), run this command from the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.classical.demo_metrics
```

The command reads `data/input.jpg`, resizes it to 128x128, and tests transmission with and without Hamming at 5, 8, 10 and 12 dB. It saves received JPEGs, decodable reconstructions, the reference image, and `metriques.csv` (BER, PSNR and SSIM) in `results/classical/fec/`. Existing files with the same names in that output folder are overwritten.

To use the same picture as the default JSCC test, set `entree` in `src/classical/pipeline.py` to `project_path("data/test_images/test.png")` before running the command. The default classical input remains `data/input.jpg`.

Other classical commands (with the virtual environment activated):

```bash
python -m scripts.shared.demo_awgn
python -m scripts.classical.demo_ber
python -m scripts.classical.demo_image
python -m scripts.classical.demo_fec
python -m scripts.classical.demo_metrics
```

`demo_image` uses JPEG quality 75 and 20 dB SNR. `demo_fec` and `demo_metrics` compare uncoded and Hamming-coded transmission at 5, 8, 10 and 12 dB. Their shared output directory is `results/classical/fec/`; `demo_metrics` also writes `metriques.csv`. JPEG decoding can fail when received bits contain errors; image metrics are calculated only for decodable JPEGs. These original behaviors are preserved. Hamming corrects one bit per 7-bit block and adds 75% transmission overhead; the comparison is at equal SNR, not equal rate.

## JSCC training, inference and evaluation

To test the trained JSCC model on a picture, run this command from the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.jscc.evaluate_jscc
```

The command uses `data/test_images/test.png` and the trained checkpoint at `results/jscc/best_model.pth`. It saves the reference image, reconstructions at 0, 5, 10, 15 and 20 dB, and `metrics.csv` in `results/jscc/evaluation/`. Existing files with the same names in that output folder are overwritten.

Other JSCC commands (with the virtual environment activated):

```bash
python -m scripts.jscc.train_jscc
python -m scripts.jscc.evaluate_jscc
python -m scripts.jscc.evaluate_jscc_multi
python -m scripts.jscc.plot_jscc
```

- Training reads `data/train_images/`, uses the original 80/20 split, seed 42, batch size 4, 10 epochs, learning rate 0.001 and training SNR 10 dB. It writes `results/jscc/best_model.pth` and `history.csv`.
- Single-image inference loads the checkpoint and `data/test_images/test.png`, evaluates SNRs 0, 5, 10, 15 and 20 dB, and writes `results/jscc/evaluation/`.
- Multi-image evaluation retains its original input directory, `data/test_images/`, and five noise repetitions per SNR. The separate `data/test_images_multi/` dataset is also preserved. To deliberately use it, set `DATA_DIR` in `src/jscc/evaluation.py` to `project_path("data/test_images_multi")`.
- Multi-image CSVs go to `results/jscc/evaluation_multi/`; figures go to `results/jscc/graphs/graph2/`. Statistical error bars remain the sample standard deviation across image means after averaging noise repetitions.
- Plotting reads saved history and single-image evaluation outputs and writes `results/jscc/graphs/graph1/`.

Single-image metrics retain their original PNG quantization and `data_range=255`; multi-image metrics retain floating point pixels and `data_range=1.0`.

Reusable components can be imported without importing the training script:

```python
from src.jscc.model import SyntaxicJSCC
from src.jscc.dataset import ImageDataset
from src.classical.hamming74 import encode_hamming74, decode_hamming74
from src.classical.pipeline import transmettre_qpsk, sauver_jpeg
from src.channels.awgn import awgn
```

## Compatibility and paths

All original commands still work, including `python -m scripts.train_jscc`, `python -m scripts.evaluate_jscc_multi`, `python -m scripts.demo_metrics` and the other original script names. Original imports such as `from scripts.train_jscc import SyntaxicJSCC` resolve to the new implementation. Original script files also support direct execution, for example `python scripts/evaluate_jscc_multi.py`.

Use `python -m ...` for the new grouped entry points. Default data/results paths are now anchored to this project root, rather than the current working directory. Architecture definitions, tensor calculations, checkpoint keys, random seeds, image preprocessing and experimental settings are unchanged.

## Preserved results and experiments

Every existing dataset, trained model, checkpoint, CSV, JPEG, reconstruction and plot is retained. Only the following result locations changed:

| Previous location | Current location |
| --- | --- |
| `results/constellation_*dB.png` | `results/classical/constellations/` |
| Root-level JPEGs and PNGs in `results/` | `results/classical/image/` |
| `results/fec/` | `results/classical/fec/` |
| Root plots in `results/graphs_jscc/` | `results/jscc/graphs/graph1/` |
| `results/graphs_jscc/graph2/` | `results/jscc/graphs/graph2/` |
| `results/jscc/` model, history and evaluations | Same paths |

The manifest records every original file and its current location. Original Python sources and README are archived under `docs/`. The virtual environment and Git metadata were left in place. Nothing was pushed to GitHub.

Experiment commands keep their original overwrite behavior for fixed output filenames. Before a new training/evaluation run, copy any results you want to retain to a separate experiment directory. Verification below uses temporary outputs and preserves saved experiments.

## Verification

```bash
python -m unittest discover -s tests -v
python -m scripts.verify_reorganization
python -m scripts.jscc.test_encoder
python -m scripts.jscc.test_norm
python -m scripts.jscc.test_awgn_jscc
python -m scripts.jscc.test_decoder
```

The unit checks cover AWGN SNR and gradients, artifact hashes, legacy imports, unchanged function/class bodies, checkpoint loading, dataset loading, Hamming correction and exact seeded forward/Adam-step equivalence with the original model. Interpreter caches are excluded from preservation checks because Python regenerates them.

The smoke verifier runs all 13 experiment/diagnostic implementations in temporary output folders. Its training check uses one epoch on five existing images; it does not rerun the full training experiment or replace the saved checkpoint.

## Channel convention

`SNR = E[|z|^2] / E[|n|^2]`, with `y = z + n` and complex Gaussian noise `n = nr + j ni`. Each real noise component has variance `P_signal / (2 * SNR_linear)`. The simulation operates on complex baseband symbols.
