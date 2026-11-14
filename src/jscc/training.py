
"""
PROJET 07 — Communication Sémantique Images et Vidéos
INSAT — 2026/2027

ÉTAPE 5 : ENTRAÎNEMENT DU JSCC SYNTAXIQUE

Objectif :
    Entraîner conjointement l'encodeur et le décodeur CNN
    pour reconstruire des images transmises sur un canal AWGN.

Architecture :
    Image RGB (128x128)
        -> Encodeur CNN
        -> 4096 valeurs latentes
        -> 2048 symboles complexes normalisés
        -> Canal AWGN
        -> Décodeur CNN
        -> Image reconstruite

Apprentissage :
    Fonction de perte : MSE
    Optimiseur : Adam
    SNR entraînement : 10 dB
    Epochs : 10
    Batch size : 4
    Device : CPU

Données :
    data/train_images/

Sorties :
    results/jscc/best_model.pth
    results/jscc/history.csv

Exécution :
    python -m scripts.train_jscc
"""

import csv
import random
from pathlib import Path
from src.paths import project_path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from src.jscc.encoder import EncoderCNN
from src.jscc.decoder import DecoderCNN
from src.jscc.normalization import PowerNormalization
from src.channels.awgn import awgn


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = project_path("data/train_images")
RESULTS_DIR = project_path("results/jscc")

IMAGE_SIZE = 128
SEED = 42
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_SNR_DB = 10.0

DEVICE = torch.device("cpu")




# ============================================================
# 2. DATASET
# ============================================================

from src.jscc.dataset import ImageDataset


# ============================================================
# 3. MODÈLE JSCC COMPLET
# ============================================================

from src.jscc.model import SyntaxicJSCC


# ============================================================
# 4. ENTRAÎNEMENT
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0
    total_images = 0

    for images in loader:

        images = images.to(DEVICE)

        # Réinitialiser les gradients.
        optimizer.zero_grad()

        # Propagation avant.
        reconstructed = model(
            images,
            TRAIN_SNR_DB
        )

        # Erreur de reconstruction.
        loss = criterion(
            reconstructed,
            images
        )

        # Backpropagation.
        loss.backward()

        # Mise à jour Adam.
        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )
        total_images += images.size(0)

    return total_loss / total_images


# ============================================================
# 5. VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_images = 0

    for images in loader:

        images = images.to(DEVICE)

        reconstructed = model(
            images,
            TRAIN_SNR_DB
        )

        loss = criterion(
            reconstructed,
            images
        )

        total_loss += (
            loss.item() * images.size(0)
        )
        total_images += images.size(0)

    return total_loss / total_images


# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 55)
    print("PROJET 07 — ENTRAÎNEMENT JSCC SYNTAXIQUE")
    print("=" * 55)

    print(f"Device : {DEVICE}")

    # --------------------------------------------------------
    # Charger le dataset
    # --------------------------------------------------------

    dataset = ImageDataset(DATA_DIR)

    if len(dataset) < 5:
        raise ValueError(
            "Ajoute au moins 5 images pour "
            "séparer entraînement et validation."
        )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Images totales : {len(dataset)}")
    print(f"Entraînement : {train_size}")
    print(f"Validation : {val_size}")

    # --------------------------------------------------------
    # Créer le modèle
    # --------------------------------------------------------

    model = SyntaxicJSCC().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    nb_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Paramètres entraînables : {nb_params}")
    print(f"SNR entraînement : {TRAIN_SNR_DB} dB")
    print(f"Epochs : {EPOCHS}")

    # --------------------------------------------------------
    # Boucle d'entraînement
    # --------------------------------------------------------

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
        })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train MSE = {train_loss:.6f} | "
            f"Val MSE = {val_loss:.6f}"
        )

        # Sauvegarder le meilleur modèle.
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "epoch": epoch,
                "val_mse": val_loss,
                "train_snr_db": TRAIN_SNR_DB,
                "image_size": IMAGE_SIZE,
                "cbr": 0.125,
            }, RESULTS_DIR / "best_model.pth")

    # --------------------------------------------------------
    # Enregistrer l'historique
    # --------------------------------------------------------

    history_path = RESULTS_DIR / "history.csv"

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_mse",
                "val_mse"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 55)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 55)

    print(f"Meilleur MSE validation : {best_val_loss:.6f}")
    print(f"Modèle sauvegardé : {RESULTS_DIR / 'best_model.pth'}")
    print(f"Historique : {history_path}")


if __name__ == "__main__":
    main()
