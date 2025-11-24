import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import Dataset
import logging
from util import get_logger
logger = get_logger(__name__)
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import joblib

from data.patch_feature_extractor import PatchFeatureExtractor

class RandomForestFeatureDataset(Dataset):
    """
    Efficient dataset for Random Forest with batch processing and caching.
    Separates concerns and implements memory-efficient patch extraction.
    """
    
    def __init__(self, 
                 image_tuple_dataset,
                 patch_size: int = 13,
                 patches_per_class: int = 75000,
                 all_patches: bool = False,
                 augment: bool = True,
                 balance: bool = False,
                 cache_dir: Optional[str] = None,
                 n_workers: int = 4,
                 scaler=None,
                 fit_scaler: bool = True):
        
        self.image_tuple_dataset = image_tuple_dataset
        self.patch_size = patch_size
        self.patches_per_class = patches_per_class
        self.all_patches = all_patches
        self.augment = augment
        self.balance = balance
        self.scaler = scaler or StandardScaler()
        self.cache_dir = cache_dir
        self.n_workers = n_workers
        
        self.feature_extractor = PatchFeatureExtractor()
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.features, self.labels = self._extract_features_and_labels()
        if fit_scaler:
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.features = self.scaler.transform(self.features)
        logger.info(f"Dataset created with {len(self.features)} samples")
    
    def _get_cache_path(self) -> Optional[str]:
        """Generate cache file path based on parameters"""
        if not self.cache_dir:
            return None
        
        cache_name = f"features_p{self.patch_size}_n{self.patches_per_class}_b{self.augment}.pkl"
        return os.path.join(self.cache_dir, cache_name)
    
    def _load_from_cache(self) -> Optional[Tuple[np.ndarray, List]]:
        """Load features from cache if available"""
        cache_path = self._get_cache_path()
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading features from cache: {cache_path}")
            try:
                return joblib.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, features: np.ndarray, labels: List):
        """Save features to cache"""
        cache_path = self._get_cache_path()
        if cache_path:
            logger.info(f"Saving features to cache: {cache_path}")
            try:
                joblib.dump((features, labels), cache_path)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def _extract_patches_batch(self, images: List, labels: List, masks: List) -> Tuple[np.ndarray, np.ndarray]:
        """Extract patches in batches for efficiency"""
        patches, patch_labels, _ = self.feature_extractor.extract_patches(
            images, labels, masks,
            patch_size=self.patch_size,
            patches_total=self.patches_per_class,
            all_patches=self.all_patches
        )
        return patches, patch_labels
    
    def _compute_features_parallel(self, patches: List) -> np.ndarray:
        """Compute features in parallel for better performance"""
        def compute_single_feature(patch):
            return self.feature_extractor.extract_features(patch)
        
        logger.info(f"Computing features for {len(patches)} patches using {self.n_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            features = list(executor.map(compute_single_feature, patches))
        
        return np.array(features)
    
    def _extract_features_and_labels(self) -> Tuple[np.ndarray, List]:
        """Extract features and labels with caching and parallel processing"""

        cached_data = self._load_from_cache()
        if cached_data is not None:
            return cached_data

        images, labels, masks = [], [], []
        
        logger.info(f"Loading {len(self.image_tuple_dataset)} images...")
        for i in range(len(self.image_tuple_dataset)):
            _, img, mask, label = self.image_tuple_dataset[i]
            images.append(img)
            labels.append(label)
            masks.append(mask)

        logger.info("Extracting patches...")
        patches, patch_labels = self._extract_patches_batch(images, labels, masks)

        patch_labels_array = np.array(patch_labels)
        n_positive = np.sum(patch_labels_array == 1)
        n_negative = np.sum(patch_labels_array == 0)

        if self.balance:
            logger.info(f"Extracted {len(patches)} patches:")
            logger.info(f"  - Positive patches (vessels): {n_positive}")
            logger.info(f"  - Negative patches (background): {n_negative}")
            logger.info(f"Balancing enabled.")
            min_count = min(n_positive, n_negative)
            
            positive_indices = np.where(patch_labels_array == 1)[0]
            negative_indices = np.where(patch_labels_array == 0)[0]
            
            balanced_indices = np.concatenate([
            np.random.choice(positive_indices, min_count, replace=False),
            np.random.choice(negative_indices, min_count, replace=False)
            ])
            
            patches = [patches[i] for i in balanced_indices]
            patch_labels = [patch_labels[i] for i in balanced_indices]
            
            patch_labels_array = np.array(patch_labels)
            n_positive = np.sum(patch_labels_array == 1)
            n_negative = np.sum(patch_labels_array == 0)
        
        logger.info(f"Extracted {len(patches)} patches:")
        logger.info(f"  - Positive patches (vessels): {n_positive}")
        logger.info(f"  - Negative patches (background): {n_negative}")
        logger.info(f"  - Balance ratio: {n_positive/(n_positive+n_negative):.3f}")

        if self.augment and n_positive > 0:
            patches, patch_labels = self._augment_vessel_patches(patches, patch_labels)

        features = self._compute_features_parallel(patches)
        
        logger.info(f"Feature matrix shape: {features.shape}")
        
        self._save_to_cache(features, patch_labels)
        
        return features, patch_labels
    
    def _augment_vessel_patches(self, patches: List, labels: List) -> Tuple[List, List]:
        """Apply simple augmentation to vessel patches"""
        augmented_patches = []
        augmented_labels = []
        
        for patch, label in zip(patches, labels):
            augmented_patches.append(patch)
            augmented_labels.append(label)
            
            # Augment only the vessel patches (label == 1)
            #if label == 1:
            flipped_h = np.fliplr(patch)
            augmented_patches.append(flipped_h)
            augmented_labels.append(label)
            
            flipped_v = np.flipud(patch)
            augmented_patches.append(flipped_v)
            augmented_labels.append(label)
    
        logger.info(f"Augmentation increased dataset from {len(patches)} to {len(augmented_patches)} patches")
        return augmented_patches, augmented_labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all features and labels as arrays"""
        return self.features, np.array(self.labels)
