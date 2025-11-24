from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
import cv2
import csv
from skimage.measure import moments_central
from scipy.ndimage import convolve, sobel
from util import get_logger
logger = get_logger(__name__)

class PatchFeatureExtractor:
    def extract_patches(self, images: list, labels: list, masks: list, patch_size: int = 27, patches_total: int = 10000, all_patches: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Extracts a balanced number of patches from each class (1 and 0) per image or all valid patches.
        Expects images and masks in (C, H, W) ndarray format.
        Returns arrays of patches and their corresponding labels.
        """

        patches = []
        patch_labels = []
        patch_coords = []

        half_patch = patch_size // 2
        kernel = np.ones((patch_size, patch_size), dtype=np.uint8)

        for img_idx, (img, label, mask) in enumerate(zip(images, labels, masks)):
            logger.debug(f"Processing image {img_idx + 1}/{len(images)}")
            positive_idx = []
            negative_idx = []

            # img: (C, H, W), mask: (C, H, W)
            # Use first channel for mask if multi-channel
            if mask.ndim == 3:
                mask_gray = (mask[0] > 0).astype(np.uint8)
            else:
                mask_gray = (mask > 0).astype(np.uint8)

            logger.debug(f"  Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
            logger.debug(f"  Mask_gray shape: {mask_gray.shape}, unique values: {np.unique(mask_gray)}")

            if not np.any(mask_gray):
                logger.warning(f"Mask for image {img_idx} is all zeros - skipping")
                continue

            coverage = convolve(mask_gray, kernel, mode='constant', cval=0)
            patch_area = patch_size * patch_size
            valid_mask = coverage >= (patch_area * 0.2)

            logger.debug(f"  Valid mask regions: {np.sum(valid_mask)} out of {valid_mask.size}")

            if label.ndim == 3:
                label_2d = label[0]
            else:
                label_2d = label

            logger.debug(f"  Label shape: {label.shape}, unique values: {np.unique(label)}")
            logger.debug(f"  Label_2d shape: {label_2d.shape}, unique values: {np.unique(label_2d)}")

            valid_patch_count = 0
            for y in range(half_patch, label_2d.shape[0] - half_patch):
                for x in range(half_patch, label_2d.shape[1] - half_patch):
                    if valid_mask[y, x]:
                        valid_patch_count += 1
                        label_val = label_2d[y, x]
                        if label_val == 1 or label_val == 255:
                            positive_idx.append((y, x))
                        elif label_val == 0:
                            negative_idx.append((y, x))

            logger.debug(f"  Valid patches found: {valid_patch_count}")
            logger.debug(f"  Positive candidates: {len(positive_idx)}")
            logger.debug(f"  Negative candidates: {len(negative_idx)}")

            if len(positive_idx) == 0 and len(negative_idx) == 0:
                logger.warning(f"No valid patches found for image {img_idx}")
                continue

            np.random.shuffle(positive_idx)
            np.random.shuffle(negative_idx)
            
            if all_patches:
                num_pos = len(positive_idx)
                num_neg = len(negative_idx)
            else:
                frac_pos = len(positive_idx) / (len(positive_idx) + len(negative_idx))
                frac_neg = 1 - frac_pos

                num_pos = min(len(positive_idx), int(patches_total * frac_pos))
                num_neg = min(len(negative_idx), int(patches_total * frac_neg))

            selected_pos = positive_idx[:num_pos]
            selected_neg = negative_idx[:num_neg]

            logger.debug(f"  Selected positive patches: {len(selected_pos)}")
            logger.debug(f"  Selected negative patches: {len(selected_neg)}")

            for y, x in selected_pos:
                patch = img[:, y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
                # (C, H, W) patch to (H, W, C) for feature extraction
                patches.append(np.transpose(patch, (1, 2, 0)))
                patch_coords.append((y, x))
                patch_labels.append(1)

            for y, x in selected_neg:
                patch = img[:, y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
                patches.append(np.transpose(patch, (1, 2, 0)))
                patch_coords.append((y, x))
                patch_labels.append(0)

        logger.info(f"Total patches extracted: {len(patches)}")
        return np.array(patches), np.array(patch_labels), patch_coords
    
    def extract_features(self, patch):
        # patch: (H, W, C) expected
         # Color features
        color_vars = self.color_variance(patch)
        central_moms = self.central_moments(patch)
        hu_moms = self.hu_moments(patch)
        # Texture features (GLCM)
        glcm_feats = self.glcm_features(patch)  # e.g., contrast, homogeneity, energy
        # Gradient features
        grad_feats = self.gradient_features(patch)
        # Neighborhood statistics
        neigh_mean, neigh_std = self.neighborhood_stats(patch, size=5)
        
        features = np.concatenate([
            color_vars, central_moms, hu_moms,
            glcm_feats, grad_feats,
            np.atleast_1d(neigh_mean), np.atleast_1d(neigh_std)
        ])
        return features
    
    def glcm_features(self, patch):
        # Accepts (H, W) or (H, W, 1) with values in [0, 1]
        if patch.ndim == 3 and patch.shape[2] == 1:
            gray = patch[..., 0]
        elif patch.ndim == 2:
            gray = patch
        else:
            raise ValueError("Patch must be 2D or 3D with 1 channel (H, W) or (H, W, 1)")
        gray = (gray * 255).astype(np.uint8)
        gray = (gray // 16).astype(np.uint8)  # Quantize to 16 levels

        glcm = graycomatrix(
            gray,
            distances=[1, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=16,
            symmetric=True
        )

        features = []
        for prop in ['contrast', 'homogeneity', 'energy']:
            vals = graycoprops(glcm, prop)
            features.extend([np.mean(vals), np.std(vals)])

        return np.array(features)

    def gradient_features(self, patch, bins=8):
        # Accepts (H, W) or (H, W, 1)
        if patch.ndim == 3 and patch.shape[2] == 1:
            gray = patch[..., 0]
        elif patch.ndim == 2:
            gray = patch
        else:
            raise ValueError("Patch must be 2D or 3D with 1 channel (H, W) or (H, W, 1)")
        gray = gray.astype(np.float32)

        # Sobel gradients
        dx = sobel(gray, axis=1)
        dy = sobel(gray, axis=0)

        # Magnitude and orientation
        mag = np.hypot(dx, dy)
        ori = np.arctan2(dy, dx)  # [-π, π]

        # 8-bin histogram (0-360°)
        ori_deg = np.rad2deg(ori) % 360
        hist, _ = np.histogram(ori_deg, bins=bins, range=(0, 360), weights=mag)
        hist = hist / (np.sum(hist) + 1e-8)

        return np.concatenate([
            [np.mean(mag), np.std(mag)],
            hist.flatten()
        ])

    def neighborhood_stats(self, patch, size=5):
        # Center coordinates
        h, w = patch.shape[:2]
        cx, cy = h // 2, w // 2
        half = size // 2
        # Extract neighborhood window, handle borders
        x1, x2 = max(cx - half, 0), min(cx + half + 1, h)
        y1, y2 = max(cy - half, 0), min(cy + half + 1, w)
        if patch.ndim == 3:
            region = patch[x1:x2, y1:y2, :]
            mean = np.mean(region, axis=(0, 1))
            std = np.std(region, axis=(0, 1))
        else:
            region = patch[x1:x2, y1:y2]
            mean = np.mean(region)
            std = np.std(region)
        return mean, std

    # nie działa dla extract_features
    def save_to_csv(self, path: str, patches: np.ndarray, labels: np.ndarray):
        num_features = patches.shape[1] * patches.shape[2] * (patches.shape[3] if patches.ndim == 4 else 1)
        with open(path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [f'feature{i+1}' for i in range(num_features)] + ['label']
            writer.writerow(header)
            for patch, label in zip(patches, labels):
                flat_patch = patch.flatten()
                row = flat_patch.tolist() + [label]
                writer.writerow(row)
                
    def color_variance(self, patch: np.ndarray) -> np.ndarray:
        # patch: h x w x c
        # Output: c (variance for each channel)
        if patch.ndim != 3:
            raise ValueError(f"Patch should be of shape: h x w x c. Got {patch.ndim} dimensions instead.")
        return np.var(patch, axis=(0, 1))

    def central_moments(self, patch: np.ndarray) -> np.ndarray:
        # patch: h x w x c
        # Output: 3 values for each channel + 3 for gray: [m[2,0], m[1,1], m[0,2]] * (c+1)
        # For grayscale (already green channel)
        features = []
        if patch.ndim == 3 and patch.shape[2] == 1:
            gray = patch[:, :, 0].astype(np.float64)
        elif patch.ndim == 2:
            gray = patch.astype(np.float64)
        else:
            raise ValueError(f"Unexpected patch shape: {patch.shape}")

        if np.std(gray) < 1e-10:  # Essentially uniform patch
            features.extend([0.0, 0.0, 0.0])
        else:
            m_gray = moments_central(gray)
            features.extend([m_gray[2, 0], m_gray[1, 1], m_gray[0, 2]])
        return np.array(features)

    def hu_moments(self, patch: np.ndarray) -> np.ndarray:
        # Accepts (h, w) or (h, w, 1)
        features = []
        if patch.ndim == 3 and patch.shape[2] == 1:
            gray = patch[..., 0]
        elif patch.ndim == 2:
            gray = patch
        else:
            raise ValueError("Patch must be 2D or 3D with 1 channel (h x w or h x w x 1)")
        m = cv2.moments(gray)
        hu = cv2.HuMoments(m).flatten()
        features.extend(hu)
        return np.array(features)

    def extract_patches_inference(self, images: list, masks: list, patch_size: int = 27) -> tuple[np.ndarray, list]:
        """
        Extract patches for every pixel in the image (sliding window).
        No label required, just extract all possible patches.
        """
        patches = []
        patch_coords = []
        half_patch = patch_size // 2

        for img, mask in zip(images, masks):
            # Get image dimensions (assuming img is C, H, W)
            img_h, img_w = img.shape[1], img.shape[2]

            for y in range(half_patch, img_h - half_patch):
                for x in range(half_patch, img_w - half_patch):
                    patch = img[:, y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
                    patches.append(np.transpose(patch, (1, 2, 0)))  # (C,H,W) -> (H,W,C)
                    patch_coords.append((y, x))

        return np.array(patches), patch_coords
