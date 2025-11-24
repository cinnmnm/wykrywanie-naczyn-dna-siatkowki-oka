# Copied from DLUnet/Core/inference.py
from util import get_logger
logger = get_logger(__name__)
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from .model import UNet
from util import load_config_yaml 
from data.dataset_supplier import DatasetSupplier
from data.image_tuple_dataset import ImageTupleDataset


class InferencePatchDataset:
    """
    Dataset for inference that handles patch extraction and reconstruction
    with non-overlapping regions for final prediction.
    """
    def __init__(self, dataset, patch_size, stride, image_size):
        """
        Args:
            dataset: ImageTupleDataset instance
            patch_size: Size of patches (h, w)
            stride: Stride between patches (sh, sw)
            image_size: Original image size (h, w)
        """
        self.dataset = dataset
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.patch_size = patch_size
        self.stride = stride
        self.image_size = image_size        
        self.patches = []
        self.patch_positions = []
        self._extract_patches()
    
    def _extract_patches(self):
        logger.debug("Extracting patches...")
        logger.debug(f"   - Patch size: {self.patch_size}")
        logger.debug(f"   - Stride: {self.stride}")
        logger.debug(f"   - Target image size: {self.image_size}")
        
        for img_idx in range(len(self.dataset)):
            id, img, mask, label = self.dataset[img_idx]
            
            # img is already (C, H, W)
            c, h, w = img.shape
            ph, pw = self.patch_size
            sh, sw = self.stride      

            patch_count = 0
            for i in range(0, h - ph + 1, sh):
                for j in range(0, w - pw + 1, sw):
                    img_patch = img[:, i:i+ph, j:j+pw]
                    img_t = torch.from_numpy(img_patch).float()
                    self.patches.append(img_t)
                    self.patch_positions.append((img_idx, i, j))
                    patch_count += 1
            
            logger.debug(f"Extracted {patch_count} patches from image {id}")
        
        logger.debug(f"Total patches extracted: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    def reconstruct_predictions(self, predictions):
        """
        Reconstruct full image predictions from patch predictions.
        Uses only the central region of each patch (no averaging).
        For patch size 2n x 2n and stride n, the central n x n region is used.
        The final image is tiled from these central regions.
        Args:
            predictions: List of patch predictions (each should be (C, H, W))
        Returns:
            List of full image predictions, one per input image
        """
        num_images = len(self.dataset)
        h, w = self.image_size
        ph, pw = self.patch_size
        sh, sw = self.stride

        ch, cw = sh, sw
        start_h = (ph - ch) // 2
        start_w = (pw - cw) // 2

        output_images = [np.zeros((h, w), dtype=np.float32) for _ in range(num_images)]

        for pred, (img_idx, i, j) in zip(predictions, self.patch_positions):
            i += (ph - ch) // 2
            j += (pw - cw) // 2 
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
            if pred.ndim == 3:
                pred = pred[0]

            central_patch = pred[start_h:start_h+ch, start_w:start_w+cw]
            end_i = min(i + ch, h)
            end_j = min(j + cw, w)
            patch_h = end_i - i
            patch_w = end_j - j
            output_images[img_idx][i:end_i, j:end_j] = central_patch[:patch_h, :patch_w]

        return output_images


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=16):
    """
    Run inference on a dataset and return reconstructed predictions.
    
    Args:
        model: Trained model
        dataset: InferencePatchDataset
        device: torch device
        batch_size: Batch size for inference
    
    Returns:
        List of prediction maps (one per input image)
    """
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_predictions = []
    
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        outputs = torch.sigmoid(outputs)
        
        for i in range(outputs.shape[0]):
            all_predictions.append(outputs[i])

    return dataset.reconstruct_predictions(all_predictions)


def main(cfg, model_path, output_dir=None):
    """
    Run inference with a trained model.

    Args:
        cfg: Configuration dictionary
        model_path: Path to trained model weights
        output_dir: Directory to save predictions (optional)
    """
    train_cfg = cfg["DLUnet"]

    device_name = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = UNet(
        train_cfg["n_channels"],
        train_cfg["n_classes"],
        base_c=train_cfg.get("base_c", 64),
        dropout_rate=train_cfg.get("dropout_rate", 0.0)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Successfully loaded model from: {model_path}")
    model.to(device)

    data_tuples = DatasetSupplier.get_dataset(cfg["image_dir"])

    resize_shape = train_cfg.get("resize_shape", cfg.get("resize_shape", (256, 256)))
    image_dataset = ImageTupleDataset(data_tuples, resize_shape)
    
    inference_dataset = InferencePatchDataset(
        image_dataset,
        train_cfg["patch_size"], 
        train_cfg["stride"],
        resize_shape  
    )

    logger.info(f"Running inference on {len(data_tuples)} images with {len(inference_dataset)} patches")
    predictions = run_inference(model, inference_dataset, device, batch_size=train_cfg.get("batch_size", 16))
    
    output_dir = cfg.get("output_dir", output_dir) # TEMP
    output_dir = output_dir + "/dlunet"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, (pred, (id, _, _, _)) in enumerate(zip(predictions, data_tuples)):
            pred_uint8 = (pred * 255).astype(np.uint8)
            import cv2
            cv2.imwrite(os.path.join(output_dir, f"{id}_prediction.png"), pred_uint8)
        logger.info(f"Saved predictions to {output_dir}")
    
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run U-Net inference")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--model", default="checkpoints/unet/unet_best.pt", help="Path to trained model weights")
    parser.add_argument("--output", help="Output directory for predictions")
    args = parser.parse_args()
    
    cfg = load_config_yaml(args.config)
    predictions = main(cfg, args.model, args.output)
    logger.info(f"Generated {len(predictions)} predictions")
