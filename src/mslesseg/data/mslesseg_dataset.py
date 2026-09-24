from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


MODALITIES = ["T1", "T2", "FLAIR"]
PATCH_SIZE = (112, 160, 128)


class MSLesSegDataset(Dataset):
    """
    PyTorch dataset for MSLesSeg.

    Each sample contains:
        image: [3, D, H, W]
               channels = T1, T2, FLAIR

        mask:  [1, D, H, W]
               binary lesion mask

    During training, a random 112x160x128 patch is extracted.
    During validation/test, a deterministic center patch is used.
    """

    def __init__(
        self,
        split_csv,
        processed_dir="data/processed/mslesseg",
        patch_size=PATCH_SIZE,
        training=False,
    ):
        self.split_csv = Path(split_csv)
        self.processed_dir = Path(processed_dir)
        self.patch_size = tuple(patch_size)
        self.training = training

        self.df = pd.read_csv(self.split_csv)

        if len(self.df) == 0:
            raise ValueError(f"No samples found in {self.split_csv}")

    def __len__(self):
        return len(self.df)

    def _load_case(self, row):
        patient = str(row["patient"])
        timepoint = str(row["timepoint"])

        case_dir = self.processed_dir / patient / timepoint

        images = []

        for modality in MODALITIES:
            path = case_dir / f"{modality}.npy"

            if not path.exists():
                raise FileNotFoundError(f"Missing modality: {path}")

            image = np.load(path).astype(np.float32)
            images.append(image)

        mask_path = case_dir / "MASK.npy"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        mask = np.load(mask_path).astype(np.float32)

        image = np.stack(images, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return image, mask

    def _get_random_start(self, shape):
        starts = []

        for size, patch in zip(shape, self.patch_size):
            if size < patch:
                raise ValueError(
                    f"Patch size {self.patch_size} is larger than "
                    f"image shape {shape}"
                )

            max_start = size - patch

            if max_start == 0:
                starts.append(0)
            else:
                starts.append(np.random.randint(0, max_start + 1))

        return starts

    def _get_center_start(self, shape):
        starts = []

        for size, patch in zip(shape, self.patch_size):
            if size < patch:
                raise ValueError(
                    f"Patch size {self.patch_size} is larger than "
                    f"image shape {shape}"
                )

            starts.append((size - patch) // 2)

        return starts

    def _crop_patch(self, image, mask):
        spatial_shape = image.shape[1:]

        if self.training:
            starts = self._get_random_start(spatial_shape)
        else:
            starts = self._get_center_start(spatial_shape)

        d, h, w = starts
        pd, ph, pw = self.patch_size

        image = image[
            :,
            d:d + pd,
            h:h + ph,
            w:w + pw,
        ]

        mask = mask[
            :,
            d:d + pd,
            h:h + ph,
            w:w + pw,
        ]

        return image, mask

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image, mask = self._load_case(row)

        image, mask = self._crop_patch(image, mask)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        mask = (mask > 0).float()

        return {
            "image": image,
            "mask": mask,
            "patient": str(row["patient"]),
            "timepoint": str(row["timepoint"]),
        }


if __name__ == "__main__":
    train_dataset = MSLesSegDataset(
        "data/processed/splits/train.csv",
        training=True,
    )

    sample = train_dataset[0]

    print("Dataset samples:", len(train_dataset))
    print("Image shape:", tuple(sample["image"].shape))
    print("Mask shape:", tuple(sample["mask"].shape))
    print("Image dtype:", sample["image"].dtype)
    print("Mask dtype:", sample["mask"].dtype)
    print("Patient:", sample["patient"])
    print("Timepoint:", sample["timepoint"])
    print("Image contains NaN:", torch.isnan(sample["image"]).any().item())
    print("Mask values:", torch.unique(sample["mask"]).tolist())