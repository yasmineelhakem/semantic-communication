import torch.nn as nn

class SemanticEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            # First convolution
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # Second convolution
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # Produce two channels: I and Q
            nn.Conv2d(
                in_channels=64,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return self.encoder(x)
