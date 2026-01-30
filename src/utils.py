import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import the logic from your modular dataset
from src.dataset import CityscapesKaggleDataset

def bake_masks(input_raw_dir, output_processed_dir):
    """
    Bakes raw JPG pairs into single-channel encoded PNGs.
    input_raw_dir: Path to raw cityscapes_data (contains train/val)
    output_processed_dir: Where to save the single-channel PNGs
    """
    splits = ['train', 'val']
    
    # Use the logic directly from your dataset class to ensure mapping consistency
    ds_logic = CityscapesKaggleDataset(root_dir=input_raw_dir)

    for split in splits:
        input_split_path = os.path.join(input_raw_dir, split)
        output_split_path = os.path.join(output_processed_dir, split)
        os.makedirs(output_split_path, exist_ok=True)

        files = [f for f in os.listdir(input_split_path) if f.endswith('.jpg')]
        print(f"🔥 Baking {len(files)} masks for '{split}' split...")

        for f in tqdm(files):
            img_path = os.path.join(input_split_path, f)
            combined = Image.open(img_path).convert("RGB")
            
            # Use PIL for splitting to be safer with different resolutions
            w, h = combined.size
            mask_raw = combined.crop((w // 2, 0, w, h))
            
            # Encode using your Manhattan distance logic
            encoded_mask = ds_logic._encode_mask(np.array(mask_raw))
            
            # Save as 8-bit single channel PNG
            mask_img = Image.fromarray(encoded_mask.astype(np.uint8))
            mask_img.save(os.path.join(output_split_path, f.replace('.jpg', '.png')))

if __name__ == "__main__":
    # Local Default: Adjust these if you run this on your Mac
    RAW_DIR = os.path.join(_project_root, 'data', 'raw', 'cityscapes_data')
    PROC_DIR = os.path.join(_project_root, 'data', 'processed', 'cityscapes_data')
    
    bake_masks(RAW_DIR, PROC_DIR)