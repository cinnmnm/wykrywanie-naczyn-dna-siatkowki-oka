from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import random_split
from util import load_config_yaml, get_logger
logger = get_logger(__name__)
import os
import random

class DatasetSupplier:
    @staticmethod
    def get_dataset(image_dir: str):
        """
        Returns a dataset based on the provided config path or config dict.
        """
        images_path = os.path.join(image_dir, "pictures")
        manual_path = os.path.join(image_dir, "manual")
        mask_path = os.path.join(image_dir, "mask")

        dataset = []

        for img_filename in os.listdir(images_path):
            base_name, img_ext = os.path.splitext(img_filename)
            
            potential_manual_exts = ['.gif', '.tif', '.png'] 
            actual_manual_file = None
            for ext in potential_manual_exts:
                potential_manual_name = base_name + ext
                if os.path.exists(os.path.join(manual_path, potential_manual_name)):
                    actual_manual_file = os.path.join(manual_path, potential_manual_name)
                    break

            potential_mask_exts = ['.gif', '.png', '.tif'] 
            actual_mask_file = None
            for ext in potential_mask_exts:
                potential_mask_name = base_name + '_mask' + ext 
                if os.path.exists(os.path.join(mask_path, potential_mask_name)):
                    actual_mask_file = os.path.join(mask_path, potential_mask_name)
                    break

            img_full_path = os.path.join(images_path, img_filename)

            if actual_manual_file and actual_mask_file and os.path.exists(img_full_path):
                dataset.append((base_name, img_full_path, actual_manual_file, actual_mask_file))
            else:
                logger.warning(f"Could not find matching files for base {base_name}")
                if not os.path.exists(img_full_path):
                    logger.warning(f"  Image file missing: {img_full_path}")
                if not actual_manual_file:
                    logger.warning(f"  Manual file missing for base: {base_name} in {manual_path}")
                if not actual_mask_file:
                    logger.warning(f"  Mask file missing for base: {base_name} in {mask_path}")

        return dataset
    
    @staticmethod
    def train_val_test_split(
        data_list: List[Tuple[str,str,str,str]],
        val_split: float,
        test_split: float,
        seed: Optional[int] = None
    ) -> Tuple[List, List, List]:
        if val_split + test_split >= 1.0:
            raise ValueError("val+test splits must be <1.0")
        if seed is not None:
            random.seed(seed)
        data_copy = data_list.copy()
        random.shuffle(data_copy)
        n = len(data_copy)
        n_test = int(test_split * n)
        n_val  = int(val_split  * n)
        test   = data_copy[:n_test]
        val    = data_copy[n_test:n_test+n_val]
        train  = data_copy[n_test+n_val:]
        return train, val, test
        