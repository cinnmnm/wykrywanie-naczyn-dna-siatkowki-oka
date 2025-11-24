import torch
import torch.nn.functional as F
import random
import numpy as np


class SegmentationAugmentation:
    """Data augmentation for segmentation tasks that keeps image-mask pairs aligned"""
    
    def __init__(self, 
                 horizontal_flip_prob=0.5,
                 vertical_flip_prob=0.5,
                 rotation_prob=0.3,
                 max_rotation_degrees=15,
                 brightness_prob=0.3,
                 brightness_factor=0.2,
                 contrast_prob=0.3,
                 contrast_factor=0.2):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_prob = rotation_prob
        self.max_rotation_degrees = max_rotation_degrees
        self.brightness_prob = brightness_prob
        self.brightness_factor = brightness_factor
        self.contrast_prob = contrast_prob
        self.contrast_factor = contrast_factor
    
    def __call__(self, image, mask):
        """Apply augmentations to image-mask pair"""
        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        
        # Ensure proper shape (C, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        
        # Vertical flip
        if random.random() < self.vertical_flip_prob:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        
        # Rotation (small angles to preserve structure)
        if random.random() < self.rotation_prob:
            angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
            image = self._rotate_tensor(image, angle)
            mask = self._rotate_tensor(mask, angle)
        
        # Brightness adjustment (only for image)
        if random.random() < self.brightness_prob:
            factor = 1 + random.uniform(-self.brightness_factor, self.brightness_factor)
            image = torch.clamp(image * factor, 0, 1)
        
        # Contrast adjustment (only for image)
        if random.random() < self.contrast_prob:
            factor = 1 + random.uniform(-self.contrast_factor, self.contrast_factor)
            mean = image.mean()
            image = torch.clamp((image - mean) * factor + mean, 0, 1)
        
        return image, mask
    
    def _rotate_tensor(self, tensor, angle_degrees):
        """Rotate tensor by given angle in degrees"""
        angle_rad = np.radians(angle_degrees)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_later = True
        else:
            squeeze_later = False
        
        grid = F.affine_grid(rotation_matrix, tensor.size(), align_corners=False)
        rotated = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        
        if squeeze_later:
            rotated = rotated.squeeze(0)
        
        return rotated


def get_train_augmentation():
    """Get standard training augmentation for segmentation"""
    return SegmentationAugmentation(
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        rotation_prob=0.3,
        max_rotation_degrees=10, 
        brightness_prob=0.3,
        brightness_factor=0.15,
        contrast_prob=0.3,
        contrast_factor=0.15
    )


def get_validation_augmentation():
    """No augmentation for validation"""
    return None
