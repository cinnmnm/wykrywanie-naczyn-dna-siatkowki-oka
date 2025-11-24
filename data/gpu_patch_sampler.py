import torch
from torch.utils.data import Dataset
import numpy as np
import concurrent.futures
import time
import random
from util import get_logger
logger = get_logger(__name__)

class GPUPatchSampler(Dataset):
    def __init__(self, gpu_dataset, patch_size=27, stride=1, mask_threshold=0.5, balance=False, class_ratio=1.0, max_patches_per_class=None, precomputed_indices=None, transform=None):
        logger.info(f"GPUPatchSampler: Initializing (patch_size={patch_size}, stride={stride}, mask_threshold={mask_threshold}, balance={balance})")
        self.dataset = gpu_dataset
        self.patch_size = patch_size
        self.stride = stride
        self.half = self.patch_size // 2
        self.mask_threshold = mask_threshold
        self.balance = balance
        self.class_ratio = class_ratio
        self.transform = transform
        self.max_patches_per_class = max_patches_per_class
        if precomputed_indices is not None:
            self.valid_indices = precomputed_indices
        else:
            logger.info("GPUPatchSampler: Precomputing valid indices by class")
            self.indices_class0, self.indices_class1 = self._precompute_indices_by_class(mask_threshold)
            logger.info(f"GPUPatchSampler: Found {len(self.indices_class0)} class 0 and {len(self.indices_class1)} class 1 patch centers")
            if self.balance:
                n1 = len(self.indices_class1)
                n0 = int(n1 * self.class_ratio)
                if self.max_patches_per_class:
                    n1 = min(n1, self.max_patches_per_class)
                    n0 = min(n0, self.max_patches_per_class)
                indices0 = random.sample(self.indices_class0, min(n0, len(self.indices_class0))) if len(self.indices_class0) > 0 else []
                indices1 = random.sample(self.indices_class1, min(n1, len(self.indices_class1))) if len(self.indices_class1) > 0 else []
                self.valid_indices = indices0 + indices1
                random.shuffle(self.valid_indices)
                logger.info(f"GPUPatchSampler: Balanced: {len(indices0)} class 0, {len(indices1)} class 1 patches")
            else:
                self.valid_indices = self.indices_class0 + self.indices_class1
                random.shuffle(self.valid_indices)
                logger.info(f"GPUPatchSampler: Using real distribution: {len(self.valid_indices)} patches")

    def _precompute_indices_by_class(self, threshold):
        indices_class0 = []
        indices_class1 = []
        num_images = self.dataset.masks.shape[0]
        patch_size = self.patch_size
        stride = self.stride
        half = self.half
        for img_idx in range(num_images):
            mask = self.dataset.masks[img_idx]
            label = self.dataset.labels[img_idx]
            if mask.ndim == 3:
                mask_2d = mask[0]
            else:
                mask_2d = mask
            H, W = mask_2d.shape
            for y in range(half, H - half, stride):
                for x in range(half, W - half, stride):
                    if mask_2d[y, x] > threshold:
                        center_label = label[:, y, x].view(-1)[0].item()
                        if center_label > 0.5:
                            indices_class1.append((img_idx, y, x, 1))
                        else:
                            indices_class0.append((img_idx, y, x, 0))
        return indices_class0, indices_class1

    def __getitem__(self, idx):
        tup = self.valid_indices[idx]
        img_idx, y_center, x_center, _ = tup
        img = self.dataset.images[img_idx]  # (C, H, W)
        C, H, W = img.shape
        half = self.half
        y1 = y_center - half
        y2 = y_center + half + 1
        x1 = x_center - half
        x2 = x_center + half + 1
        pad_top = max(0, -y1)
        pad_left = max(0, -x1)
        pad_bottom = max(0, y2 - H)
        pad_right = max(0, x2 - W)
        if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
            img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        y1_clamped = y1 + pad_top
        y2_clamped = y2 + pad_top
        x1_clamped = x1 + pad_left
        x2_clamped = x2 + pad_left
        patch = img[:, y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        label = self.dataset.labels[img_idx, :, y_center, x_center].view(-1)[0].long()
        if idx < 3:
            logger.debug(f"GPUPatchSampler: Patch idx={idx}, img_idx={img_idx}, center=({y_center},{x_center}), patch shape={patch.shape}, label={label.item()}")
        if self.transform:
            patch = self.transform(patch)
        return patch, label

    def __len__(self):
        return len(self.valid_indices)
