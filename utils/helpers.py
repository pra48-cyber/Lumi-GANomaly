import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AnomalyTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        SUPPORTED_EXTENSIONS = ('.tif', '.bmp', '.jpg', '.png', '.jpeg')
        self.image_paths = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ])
        
        if not self.image_paths:
            raise RuntimeError(f"No valid images found in directory: {self.data_dir}")

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, img_path
        except Exception as e:
            raise IOError(f"Failed to load or process image {img_path}: {e}")

def tensor_to_pil(tensor):
    """De-normalizes a tensor from range [-1, 1] to a PIL Image."""
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return transforms.ToPILImage()(tensor.cpu())

def generate_heatmap(error_map):
    """Generates a pseudo-color JET heatmap from a normalized matrix."""
    norm_map = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def detect_temporal_window(dataset):
    """Analyzes dataset tensor dimensionality to determine temporal sequence depth."""
    try:
        sample = dataset[0]
        tensor = sample[0] if isinstance(sample, tuple) else sample
        ndims = len(tensor.shape)

        if ndims == 3:
            return 1
        elif ndims == 4:
            return tensor.shape[0]
        elif ndims == 5:
            return tensor.shape[1]
        else:
            raise ValueError(f"Unsupported tensor dimension structure: {tensor.shape}")
    except Exception as e:
        raise RuntimeError(f"Temporal window analysis failed: {e}")
