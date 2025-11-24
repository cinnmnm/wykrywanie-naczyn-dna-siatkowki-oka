import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from data.preprocessing import ImagePreprocessing

def transform_image(image):
    image = ImagePreprocessing.extract_green_channel(image)
    image = ImagePreprocessing.apply_clahe(image)
    image = ImagePreprocessing.median_filter(image, kernel_size=3)
    image = np.expand_dims(image, axis=-1)
    image = image.astype(np.float32) / 255.0 # Normalize to [0, 1]
    #ImagePreprocessing.enhance_vessels(image)
    return image

class ImageTupleDataset(Dataset):
    def __init__(self, data_tuples, image_size=None, transform=transform_image):
        """
        Args:
            data_tuples (list of tuples): Each tuple should contain (image_path, mask_path, label_path)
            image_size (tuple): Desired image size (width, height)
            transform (callable, optional): Optional transform to be applied on the image only.
        """
        self.ids = []
        self.data = []
        self.masks = []
        self.labels = []
        self.image_size = image_size
        self.transform = transform

        for (id, img_path, label_path, mask_path) in data_tuples:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)            
            if self.image_size:
                img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
                label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)

            img = np.array(img)
            mask = np.array(mask)
            label = np.array(label)

            self.ids.append(id)
            self.data.append(img)
            self.masks.append(mask)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.ids[idx]
        img = self.data[idx]
        mask = self.masks[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = img.astype(np.float32) / 255.0

        # Convert all data to PyTorch (C, H, W) format
        # Convert image to (C, H, W) format
        if img.ndim == 3:  # RGB image (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:  # Grayscale image (H, W) -> (1, H, W)
            img = np.expand_dims(img, axis=0)
        
        # Convert mask and label to (C, H, W) format - they are grayscale so (1, H, W)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # (H, W) -> (1, H, W)

        label = label.astype(np.float32) / 255.0
        label = np.expand_dims(label, axis=0)  # (H, W) -> (1, H, W)

        return id, img, mask, label