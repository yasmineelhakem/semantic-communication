import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 128


class ImageDataset(Dataset):

    def __init__(self, folder):

        self.paths = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }
        )

        if not self.paths:
            raise ValueError(
                f"Aucune image trouvée dans {folder}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        path = self.paths[index]

        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Convertir les pixels en [0, 1].
            array = (
                np.asarray(img, dtype=np.float32)
                / 255.0
            ).copy()

        # [H, W, C] -> [C, H, W]
        image = torch.from_numpy(array).permute(
            2, 0, 1
        )

        return image
