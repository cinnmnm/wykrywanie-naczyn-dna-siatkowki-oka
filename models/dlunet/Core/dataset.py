import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, image_dataset, patch_size, stride, augment: bool = False):
        """
        Args:
            image_dataset: ImageTupleDataset instance that returns (id, img, mask, label)
                          where img, mask, label are in (C, H, W) format
            patch_size (int or tuple): Size of the patch (h, w)
            stride (int or tuple): Stride between patches (sh, sw)
        """
        self.image_dataset = image_dataset
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.patch_size = patch_size
        self.stride = stride

        self.patches = []
        self._extract_patches()
        self.augment = augment

    def _extract_patches(self):
        for idx in range(len(self.image_dataset)):
            id, img, mask, label = self.image_dataset[idx]
            
            # img, mask, label are now in (C, H, W) format
            c, h, w = img.shape
            ph, pw = self.patch_size
            sh, sw = self.stride

            for i in range(0, h - ph + 1, sh):
                for j in range(0, w - pw + 1, sw):
                    # Extract patches in (C, H, W) format
                    img_patch = img[:, i:i+ph, j:j+pw]
                    mask_patch = mask[:, i:i+ph, j:j+pw]
                    label_patch = label[:, i:i+ph, j:j+pw]
                    self.patches.append((img_patch, mask_patch, label_patch))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch, mask_patch, label_patch = self.patches[idx]
        # Apply simple on-the-fly augmentations for training patches
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_patch = img_patch[:, :, ::-1].copy()
                mask_patch = mask_patch[:, :, ::-1].copy()
                label_patch = label_patch[:, :, ::-1].copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                img_patch = img_patch[:, ::-1, :].copy()
                mask_patch = mask_patch[:, ::-1, :].copy()
                label_patch = label_patch[:, ::-1, :].copy()
            # Random 90-degree rotations
            k = np.random.choice([0, 1, 2, 3])
            if k != 0:
                img_patch = np.rot90(img_patch, k, axes=(1, 2)).copy()
                mask_patch = np.rot90(mask_patch, k, axes=(1, 2)).copy()
                label_patch = np.rot90(label_patch, k, axes=(1, 2)).copy()
        img_t = torch.from_numpy(img_patch).float()
        mask_t = torch.from_numpy(mask_patch).float()
        label_t = torch.from_numpy(label_patch).float()
        return img_t, mask_t, label_t
