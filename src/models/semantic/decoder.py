import torch.nn as nn

class SemanticDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(

            # Process received representation
            nn.Conv2d(
                in_channels=2,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            # Upsample: 32x32 -> 64x64
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # Upsample: 64x64 -> 128x128
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Output RGB values between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
