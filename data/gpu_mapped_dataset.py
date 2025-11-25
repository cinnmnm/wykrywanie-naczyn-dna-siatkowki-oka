import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.preprocessing import ImagePreprocessing
import torchvision.transforms as T
from util import get_logger
logger = get_logger(__name__)

class GPUMappedDataset(Dataset):
    def __init__(self, image_dataset, device='cuda'):
        """
        Loads and preprocesses all images/masks/labels to GPU memory for efficient patch extraction.
        
        Args:
            dataset_tuples: list of (basename, image_path, label_path, mask_path)
            device: GPU device to load data onto
            scale_shape: target shape for resizing
            picture_transform: optional transform function (applied on GPU tensors)
        """
        logger.info(f"GPUMappedDataset: Initializing with {len(image_dataset)} images, device={device}")
        self.device = device
        
        logger.info("GPUMappedDataset: Loading to GPU memory")
        images_tensors = []
        masks_tensors = []
        labels_tensors = []
        for idx in range(len(image_dataset)):
            _, img, mask, label = image_dataset[idx]
            img_tensor = torch.from_numpy(img).float().to(device)
            mask_tensor = torch.from_numpy(mask).float().to(device)
            label_tensor = torch.from_numpy(label).float().to(device)
            images_tensors.append(img_tensor)
            masks_tensors.append(mask_tensor)
            labels_tensors.append(label_tensor)
        self.images = torch.stack(images_tensors)
        self.masks = torch.stack(masks_tensors)
        self.labels = torch.stack(labels_tensors)
        logger.info(f"GPUMappedDataset: Loaded shapes - Images: {self.images.shape}, Masks: {self.masks.shape}, Labels: {self.labels.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.masks[idx]