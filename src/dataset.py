import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CityscapesKaggleDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.image_filenames = [f for f in os.listdir(self.split_dir) if f.endswith('.jpg')]
        self.transform = transform
        
        # Expanded Professional Mapping (Grouping 30+ classes into 6)
        self.color_map = {
            # 1: ROAD & FLAT (Purples/Pinks)
            (128, 64, 128): 1, (244, 35, 232): 1, (250, 170, 160): 1,
            # 2: HUMAN (Reds)
            (220, 20, 60): 2, (255, 0, 0): 2,
            # 3: VEHICLE (Blues/Browns)
            (0, 0, 142): 3, (0, 0, 70): 3, (0, 60, 100): 3, (0, 0, 230): 3, (119, 11, 32): 3,
            # 4: CONSTRUCTION/BUILDINGS (Grays/Yellow-Grays)
            (70, 70, 70): 4, (102, 102, 156): 4, (190, 153, 153): 4, (150, 100, 100): 4,
            # 5: OBJECTS/SIGNS (Yellows/Brights)
            (220, 220, 0): 5, (153, 153, 153): 5, (250, 170, 30): 5, (220, 220, 0): 5,
            # 6: NATURE (Greens)
            (107, 142, 35): 6, (152, 251, 152): 6
        }

    def encode_mask(self, mask_np):
        h, w, _ = mask_np.shape
        mask_id = np.zeros((h, w), dtype=np.int64)
        min_dist = np.full((h, w), np.inf)

        for color, class_id in self.color_map.items():
            # Manhattan distance is faster for large maps than Euclidean
            dist = np.sum(np.abs(mask_np - np.array(color)), axis=-1)
            
            # Threshold of 50 catches 'lossy' JPG colors effectively
            match = (dist < 50) & (dist < min_dist)
            mask_id[match] = class_id
            min_dist[match] = dist[match]
        return mask_id

    def __getitem__(self, idx):
        img_path = os.path.join(self.split_dir, self.image_filenames[idx])
        combined_img = np.array(Image.open(img_path).convert("RGB"))
        
        # Image is still the left half of the raw JPG
        image_np = combined_img[:, :256, :]
        
        # NEW: Load the pre-processed mask from the 'processed' folder
        # Adjust the path logic here to point to data/processed/...
        mask_path = img_path.replace('raw', 'processed').replace('.jpg', '.png')
        mask_id = np.array(Image.open(mask_path))
        
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_id).long()

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.image_filenames)