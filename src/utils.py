"""Shared utilities for data preprocessing and helpers."""

import os
import sys

# Allow running from src/ or from project root
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.dataset import CityscapesKaggleDataset


def bake_masks(project_root=None):
    """Preprocess raw Cityscapes images: extract and encode masks to single-channel PNGs."""
    root = project_root or _project_root
    raw_dir = os.path.join(root, 'data', 'raw', 'cityscapes_data')
    target_dir = os.path.join(root, 'data', 'processed', 'cityscapes_data')
    splits = ['train', 'val']

    ds_logic = CityscapesKaggleDataset(root_dir=raw_dir)

    for split in splits:
        input_split_path = os.path.join(raw_dir, split)
        output_split_path = os.path.join(target_dir, split)
        os.makedirs(output_split_path, exist_ok=True)

        files = [f for f in os.listdir(input_split_path) if f.endswith('.jpg')]
        print(f"Baking {split} masks...")

        for f in tqdm(files):
            combined = np.array(Image.open(os.path.join(input_split_path, f)).convert("RGB"))
            mask_np = combined[:, 256:, :]
            encoded_mask = ds_logic.encode_mask(mask_np)
            mask_img = Image.fromarray(encoded_mask.astype(np.uint8))
            mask_img.save(os.path.join(output_split_path, f.replace('.jpg', '.png')))


if __name__ == "__main__":
    bake_masks()
