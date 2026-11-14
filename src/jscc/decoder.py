
"""
PROJET 07 — Communication Sémantique Images et Vidéos
INSAT — 2026/2027

ÉTAPE 4 : DÉCODEUR CNN DU JSCC SYNTAXIQUE

Objectif :
    Reconstruire une image RGB à partir des
    symboles complexes reçus après le canal AWGN.

Architecture :
    2048 symboles complexes
            |
    4096 valeurs réelles
            |
    Tenseur [B, 16, 16, 16]
            |
    ConvTranspose2d : 16 -> 64 | 16x16 -> 32x32
    PReLU
            |
    ConvTranspose2d : 64 -> 32 | 32x32 -> 64x64
    PReLU
            |
    ConvTranspose2d : 32 -> 3  | 64x64 -> 128x128
    Sigmoid
            |
    Image RGB reconstruite [B, 3, 128, 128]

Remarque :
    Le décodeur doit être entraîné avec l'encodeur
    pour apprendre à reconstruire les images.
"""

import torch
from torch import nn


class DecoderCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # 16x16 -> 32x32
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.PReLU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.PReLU(),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Pixels compris entre 0 et 1
            nn.Sigmoid()
        )

    def forward(self, symbols):
        """
        Entrée :
            [B, 2048] symboles complexes

        Sortie :
            [B, 3, 128, 128] image reconstruite
        """

        if not torch.is_complex(symbols):
            raise TypeError(
                "Le décodeur attend des symboles complexes."
            )

        batch_size = symbols.shape[0]

        if symbols.ndim != 2 or symbols.shape[1] != 2048:
            raise ValueError(
                "Dimensions attendues : [batch, 2048]."
            )

        # Séparer les parties réelle et imaginaire.
        z = torch.view_as_real(symbols)

        # [B, 2048, 2] -> [B, 16, 16, 16]
        z = z.reshape(batch_size, 16, 16, 16)

        # Reconstruire l'image.
        reconstructed = self.network(z)

        return reconstructed
