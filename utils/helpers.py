import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LowLightEnhancement(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def _apply_otsu(self, img_gray):
        _, otsu_map = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_map

    def _apply_adaptive_gamma(self, img_normalized):
        mean_intensity = np.mean(img_normalized)
        epsilon = 1e-9
        gamma_map = np.log(mean_intensity + epsilon) / np.log(img_normalized + epsilon)
        gamma_map = np.clip(gamma_map, 0.1, 5.0)
        return np.power(img_normalized, gamma_map)

    def __call__(self, pil_img):
        img_np = np.array(pil_img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        otsu_gray = self._apply_otsu(img_gray)
        img_otsu = cv2.cvtColor(otsu_gray, cv2.GRAY2RGB) / 255.0
        
        img_norm = img_np / 255.0
        img_gamma = self._apply_adaptive_gamma(img_norm)
        
        img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        img_ycrcb[:, :, 0] = self.clahe.apply(img_ycrcb[:, :, 0])
        img_clahe = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB) / 255.0

        w1, w2, w3 = 0.25, 0.35, 0.40
        fused_img = (w1 * img_otsu) + (w2 * img_gamma) + (w3 * img_clahe)
        fused_img = np.clip(fused_img * 255.0, 0, 255).astype(np.uint8)
        
        return Image.fromarray(fused_img)


class LumiAnomalyDataset(Dataset):
    def __init__(self, data_dir, transform=None, temporal_window=1, enable_enhancement=True):
        self.data_dir = data_dir
        self.transform = transform
        self.temporal_window = temporal_window
        self.enhancement = LowLightEnhancement() if enable_enhancement else None
        
        SUPPORTED_EXTENSIONS = ('.tif', '.bmp', '.jpg', '.png', '.jpeg')
        self.image_paths = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ])
        
        if len(self.image_paths) < self.temporal_window:
            raise RuntimeError(f"Directory {self.data_dir} has fewer frames than temporal window size {self.temporal_window}")

    def __len__(self): 
        return len(self.image_paths) - self.temporal_window + 1

    def __getitem__(self, idx):
        sequence_tensors = []
        sequence_paths = []
        
        # Pull consecutive window frames (Eq. 7 / Eq. 25)
        for t in range(self.temporal_window):
            img_path = self.image_paths[idx + t]
            sequence_paths.append(img_path)
            
            try:
                image = Image.open(img_path).convert("RGB")
                
                if self.enhancement:
                    image = self.enhancement(image)
                    
                if self.transform:
                    tensor = self.transform(image)
                else:
                    tensor = transforms.ToTensor()(image)
                    
                sequence_tensors.append(tensor)
            except Exception as e:
                raise IOError(f"Failed to process sequence frame {img_path}: {e}")
        
        # If windowing is active, stack across temporal dimensions
        if self.temporal_window > 1:
            stacked_sequence = torch.stack(sequence_tensors, dim=0) # Shape: [T, C, H, W]
            return stacked_sequence, sequence_paths
            
        return sequence_tensors[0], sequence_paths[0]


def tensor_to_pil(tensor):
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return transforms.ToPILImage()(tensor.cpu())


def generate_heatmap(error_map):
    norm_map = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def detect_temporal_window(dataset):
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
