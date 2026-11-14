import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        # Load pretrained VGG16
        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights)

        # Extract early convolutional features
        self.features = nn.Sequential(
            *list(vgg.features.children())[:9]
        )

        # Freeze VGG16 parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.features.eval()

    def forward(self, original, reconstructed):

        # Normalize images for VGG16
        original = (original - self.mean) / self.std
        reconstructed = (reconstructed - self.mean) / self.std

        # Extract feature representations
        original_features = self.features(original)

        reconstructed_features = self.features(reconstructed)

        # Compare feature representations
        loss = nn.functional.l1_loss(
            reconstructed_features,
            original_features
        )

        return loss
