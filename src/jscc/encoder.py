
"""
PROJET 07 — Communication Sémantique Images et Vidéos
INSAT — 2026/2027

ÉTAPE 1 : ENCODEUR CNN DU JSCC SYNTAXIQUE

Objectif :
    Transformer une image RGB 128x128 en une
    représentation latente compacte.

Architecture :
    Entrée : [batch, 3, 128, 128]

    Conv1 : 3  -> 32 canaux | 128x128 -> 64x64
    PReLU

    Conv2 : 32 -> 64 canaux | 64x64 -> 32x32
    PReLU

    Conv3 : 64 -> 16 canaux | 32x32 -> 16x16

    Sortie : [batch, 16, 16, 16]

La sortie contient 4096 valeurs réelles par image,
qui représenteront 2048 symboles complexes.

Note :
    Le modèle n'est pas encore entraîné.

    La normalisation, le canal AWGN et le décodeur
    seront ajoutés dans les prochaines étapes.
"""

from torch import nn


class EncoderCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Première convolution
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.PReLU(),

            # Deuxième convolution
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.PReLU(),

            # Troisième convolution
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2
            )
        )

    def forward(self, x):
        """
        Applique les convolutions à l'image.

        Entrée : [B, 3, 128, 128]
        Sortie : [B, 16, 16, 16]
        """
        return self.network(x)
