import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CityscapesKaggleDataset(Dataset):
    def __init__(self, root_dir, split='train', target_size=(512, 512)):
        # Kaggle datasets are usually in /kaggle/input/cityscapes-image-pairs/
        self.split_dir = os.path.join(root_dir, split)
        self.image_filenames = [f for f in os.listdir(self.split_dir) if f.endswith('.jpg')]
        self.target_size = target_size
        
        # 1. Image Transform (Bilinear for smooth detail)
        self.img_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(), # Scales to [0, 1]
        ])
        
        # 2. Mask Transform (Nearest to preserve class IDs)
        self.mask_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.target_size, interpolation=T.InterpolationMode.NEAREST),
        ])

        # Standard Professional Mapping
        self.color_map = {
            (128, 64, 128): 1, (244, 35, 232): 1, (250, 170, 160): 1, # Road
            (220, 20, 60): 2, (255, 0, 0): 2,                         # Human
            (0, 0, 142): 3, (0, 0, 70): 3, (0, 60, 100): 3,           # Vehicle
            (70, 70, 70): 4, (102, 102, 156): 4, (190, 153, 153): 4,  # Construction
            (220, 220, 0): 5, (153, 153, 153): 5, (250, 170, 30): 5,  # Objects
            (107, 142, 35): 6, (152, 251, 152): 6                     # Nature
        }

    def _encode_mask(self, mask_np):
        h, w, _ = mask_np.shape
        mask_id = np.zeros((h, w), dtype=np.int64)
        min_dist = np.full((h, w), np.inf)

        for color, class_id in self.color_map.items():
            dist = np.sum(np.abs(mask_np - np.array(color)), axis=-1)
            match = (dist < 50) & (dist < min_dist)
            mask_id[match] = class_id
            min_dist[match] = dist[match]
        return mask_id

    def __getitem__(self, idx):
        img_path = os.path.join(self.split_dir, self.image_filenames[idx])
        combined_img = Image.open(img_path).convert("RGB")
        w, h = combined_img.size
        
        # Split original 256x512 into two squares before upscaling
        image = combined_img.crop((0, 0, w//2, h))
        mask_raw = combined_img.crop((w//2, 0, w, h))
        
        # Apply transforms to upscale to target_size (e.g., 512x512)
        image_tensor = self.img_transform(np.array(image))
        mask_rescaled = self.mask_transform(np.array(mask_raw))
        
        # Encode the rescaled mask into 7 classes
        mask_id = self._encode_mask(np.array(mask_rescaled))
        mask_tensor = torch.from_numpy(mask_id).long()

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.image_filenames)