import os
import cv2
import numpy as np
import logging
from util import get_logger
logger = get_logger(__name__)
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import joblib
from tqdm import tqdm

from data.preprocessing import ImagePreprocessing
from .model import RandomForestModel
from data.image_tuple_dataset import ImageTupleDataset, transform_image
from data.patch_feature_extractor import PatchFeatureExtractor

class RandomForestInference:
    """
    Sliding window inference for Random Forest retinal vessel segmentation.
    Handles full image reconstruction from patch-level predictions.
    """
    
    def __init__(self, model_path: str, config: Dict):
        """
        Initialize inference module with trained model and configuration.
        
        Args:
            model_path: Path to saved Random Forest model (.pkl)
            config: Configuration dictionary containing inference parameters
        """
        self.config = config["DLRandomForest"]
        self.model_path = model_path
        self.patch_size = self.config.get("patch_size", 15)
        self.stride = self.config.get("inference_stride", 3)
        self.resize_shape = tuple(self.config.get("resize_shape", [256, 256]))
        
        self.model = self._load_model()
        self.feature_extractor = PatchFeatureExtractor()
        
        logger.info(f"Inference initialized: patch_size={self.patch_size}, stride={self.stride}")
    
    def _load_model(self) -> RandomForestModel:
        """Load trained Random Forest model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        if isinstance(model_data, dict):
            rf_model = RandomForestModel(model_data.get('config', {}))
            rf_model.model = model_data['model']
            rf_model.is_trained = True
        else:
            rf_model = RandomForestModel({})
            rf_model.model = model_data
            rf_model.is_trained = True

        if hasattr(rf_model.model, "set_params"):
            rf_model.model.set_params(verbose=0, n_jobs=-1)
        else:
            rf_model.model.verbose = 0
            rf_model.model.n_jobs = -1
        logger.info(f"Model loaded successfully from {self.model_path}")
        return rf_model
    
    def predict_image(self, config, image_path: str, mask_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on a single retinal image using sliding window approach.
        
        Args:
            image_path: Path to input retinal image
            mask_path: Optional path to ROI mask
            
        Returns:
            Tuple of (probability_map, binary_mask)
        """
        ds = ImageTupleDataset([('inference', image_path, mask_path, mask_path)], config['resize_shape'])

        patches, patch_coords = self._extract_sliding_patches(ds[0][1], ds[0][2])
        
        if len(patches) == 0:
            logger.warning("No valid patches extracted")
            return np.zeros(self.resize_shape), np.zeros(self.resize_shape)
        
        logger.info(f"Computing features for {len(patches)} patches...")
        features = np.array([self.feature_extractor.extract_features(patch.transpose(1, 2, 0)) for patch in patches])
        
        scaler_path = os.path.join(os.path.dirname(self.model_path), "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            features = scaler.transform(features)
            logger.info(f"Features normalized using scaler from {scaler_path}")
        else:
            logger.warning("No scaler found - using raw features")

        probabilities = self.model.predict_proba(features)
        
        prob_map = self._reconstruct_from_patches(
            probabilities, patch_coords, self.resize_shape
        )
        
        return prob_map, prob_map > 0.5125
    
    def _extract_sliding_patches(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract overlapping patches using sliding window approach.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            roi_mask: Optional ROI mask to limit patch extraction
            
        Returns:
            Tuple of (patches_list, coordinates_list)
        """
        patches = []
        coords = []

        if image.ndim == 2:
            h, w = image.shape
            image = np.stack([image] * 3, axis=2)
        else:
            h, w = image.shape[1], image.shape[2]

        half_patch = self.patch_size // 2

        if image.ndim == 3:
            padded = np.pad(
            image,
            ((0, 0), (half_patch, half_patch), (half_patch, half_patch)),
            mode='edge'
            )
        else:
            padded = np.pad(
            image,
            ((half_patch, half_patch), (half_patch, half_patch)),
            mode='edge'
            )

        if roi_mask is not None:
            if roi_mask.ndim == 2:
                roi_mask_padded = np.pad(
                    roi_mask,
                    ((half_patch, half_patch), (half_patch, half_patch)),
                    mode='edge'
                )
            else:
                roi_mask_padded = np.pad(
                    roi_mask,
                    ((0, 0), (half_patch, half_patch), (half_patch, half_patch)),
                    mode='edge'
                )
        else:
            roi_mask_padded = None

        for y in range(half_patch, h - half_patch, self.stride):
            for x in range(half_patch, w - half_patch, self.stride):
                if roi_mask_padded is not None:
                    if roi_mask_padded.ndim == 2:
                        mask_val = roi_mask_padded[y + half_patch, x + half_patch]
                    else:
                        mask_val = roi_mask_padded[0, y + half_patch, x + half_patch]
                    if mask_val < 0.5:
                        continue

                patch = padded[:, y:y + self.patch_size, x:x + self.patch_size]
                if patch.shape[1:3] == (self.patch_size, self.patch_size):
                    patches.append(patch)
                    coords.append((y, x))

        return patches, coords

    def _reconstruct_from_patches(self, probabilities: np.ndarray, coordinates: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct full probability map from patch predictions with overlap handling.
        
        Args:
            probabilities: Patch-level vessel probabilities
            coordinates: Patch center coordinates
            image_shape: Target image shape (H, W)
            
        Returns:
            Reconstructed probability map
        """
        prob_map = np.zeros(image_shape, dtype=np.float32)
        count_map = np.zeros(image_shape, dtype=np.float32)
        half_patch = self.patch_size // 2
        
        for prob, (y, x) in zip(probabilities, coordinates):
            y1, y2 = max(0, y-half_patch), min(image_shape[0], y+half_patch+1)
            x1, x2 = max(0, x-half_patch), min(image_shape[1], x+half_patch+1)
            
            prob_map[y1:y2, x1:x2] += prob
            count_map[y1:y2, x1:x2] += 1

        prob_map = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map!=0)
        return prob_map
    
    def predict_batch(self, config, image_paths: List[str], output_dir: str, mask_paths: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Perform batch inference on multiple images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save prediction results
            mask_paths: Optional list of ROI mask paths
            
        Returns:
            Dictionary of results for each image
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for i, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                mask_path = mask_paths[i] if mask_paths and i < len(mask_paths) else None

                prob_map, binary_mask = self.predict_image(config, img_path, mask_path)

                img_name = Path(img_path).stem
                prob_path = os.path.join(output_dir, f"{img_name}_prob.png")
                mask_path_out = os.path.join(output_dir, f"{img_name}_mask.png")

                prob_8bit = (prob_map * 255).astype(np.uint8)
                mask_8bit = (binary_mask * 255).astype(np.uint8)
                
                cv2.imwrite(prob_path, prob_8bit)
                cv2.imwrite(mask_path_out, mask_8bit)
                
                results[img_name] = {
                    'probability_map': prob_path,
                    'binary_mask': mask_path_out,
                    'vessel_pixels': np.sum(binary_mask),
                    'total_pixels': binary_mask.size,
                    'vessel_ratio': np.mean(binary_mask)
                }
                
                logger.info(f"Processed {img_name}: vessel ratio = {results[img_name]['vessel_ratio']:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results[Path(img_path).stem] = {'error': str(e)}
        
        return results
    
    def predict_dataset(self, dataset: ImageTupleDataset) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Predict vessel segmentation for all images in dataset.
        
        Args:
            dataset: ImageTupleDataset containing images to predict
            
        Returns:
            List of tuples (image_id, probability_map, binary_map) for each image
        """
        results = []
        
        for i in range(len(dataset)):
            image_id, img, mask, label = dataset[i]

            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img[0]  # (1, H, W) -> (H, W)

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            elif mask.ndim == 3:
                mask = mask.mean(axis=0)  

            patches, patch_coords = self._extract_sliding_patches(img, mask)
            
            if len(patches) == 0:
                h, w = img.shape[:2] if img.ndim == 3 else img.shape
                prob_map = np.zeros((h, w), dtype=np.float32)
                binary_map = np.zeros((h, w), dtype=np.uint8)
            else:
                features = np.array([self.feature_extractor.extract_features(patch) for patch in patches])

                prob_output = self.model.predict_proba(features)
                
                if prob_output.ndim == 2 and prob_output.shape[1] == 2:
                    probabilities = prob_output[:, 1]  
                elif prob_output.ndim == 1:
                    probabilities = prob_output
                else:
                    probabilities = prob_output.flatten() if prob_output.ndim > 1 else prob_output

                image_shape = img.shape[:2] if img.ndim == 3 else img.shape
                prob_map = self._reconstruct_from_patches(probabilities, patch_coords, image_shape)
                binary_map = (prob_map > 0.5).astype(np.uint8)
            
            results.append((image_id, prob_map, binary_map))
        
        return results
