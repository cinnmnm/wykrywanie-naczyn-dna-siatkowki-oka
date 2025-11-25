import numpy as np
from util import get_logger
logger = get_logger(__name__)
from torch.utils.data import Dataset
from data.patch_feature_extractor import PatchFeatureExtractor

class PatchFeatureDataset(Dataset):
    """
    Dataset that extracts features from patches using PatchFeatureExtractor for Random Forest training.
    Can use all, a subset, or a balanced subset of patches.
    """
    def __init__(self, image_tuple_dataset, patch_size=27, patches_per_class=10000, all_patches=False, balance=True):
        self.image_tuple_dataset = image_tuple_dataset
        self.patch_size = patch_size
        self.patches_per_class = patches_per_class
        self.all_patches = all_patches
        self.balance = balance
        self.feature_extractor = PatchFeatureExtractor()
        self.features, self.labels = self._extract_features_and_labels()

    def _extract_features_and_labels(self):
        all_images_list = []
        all_labels_list = []
        all_masks_list = []

        for i in range(len(self.image_tuple_dataset)):
            _, img_data, mask_data, label_data = self.image_tuple_dataset[i]
            all_images_list.append(img_data)
            all_labels_list.append(label_data)
            all_masks_list.append(mask_data)
        
        if not all_images_list:
            return np.array([]), np.array([])

        logger.info(f"Extracting features from {len(all_images_list)} images")

        for i, (img, mask, label) in enumerate(zip(all_images_list[:1], all_masks_list[:1], all_labels_list[:1])):
            logger.debug(f"Image {i}: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}]")
            logger.debug(f"Mask {i}: shape={mask.shape}, dtype={mask.dtype}, range=[{mask.min():.3f}, {mask.max():.3f}]")
            logger.debug(f"Label {i}: shape={label.shape}, dtype={label.dtype}, unique values={np.unique(label)}")

        raw_patches, patch_labels, patch_coords = self.feature_extractor.extract_patches(
            images=all_images_list, 
            labels=all_labels_list, 
            masks=all_masks_list,
            patch_size=self.patch_size,
            patches_total=self.patches_per_class,
            all_patches=self.all_patches
        )
        
        logger.info(f"Extracted {len(raw_patches)} patches (requested {self.patches_per_class} per class)")
        if len(patch_labels) > 0:
            n_positive = np.sum(np.array(patch_labels) == 1)
            n_negative = np.sum(np.array(patch_labels) == 0)
            logger.info(f"  - Positive patches (vessels): {n_positive}")
            logger.info(f"  - Negative patches (background): {n_negative}")
            logger.info(f"  - Balance ratio: {n_positive/(n_positive+n_negative):.3f}")
        else:
            logger.warning("No patches extracted")
        
        if raw_patches.shape[0] == 0:
            logger.warning("No patches extracted - returning empty arrays")
            return np.array([]), np.array([])

        logger.info(f"Computing features for {len(raw_patches)} patches")

        features = np.array([self.feature_extractor.extract_features(patch) for patch in raw_patches])
        logger.info(f"Features shape: {features.shape}, Labels shape: {len(patch_labels)}")
        
        return features, np.array(patch_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]