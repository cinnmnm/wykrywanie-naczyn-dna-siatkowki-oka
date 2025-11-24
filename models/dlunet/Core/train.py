import os
import json
import torch
import pathlib
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.dlunet.Core.model import UNet
from models.dlunet.Core.dataset import PatchDataset
from models.dlunet.Core.loss import DiceCEWithWeight
from util import load_config_yaml, setup_logging, get_logger
from data.dataset_supplier import DatasetSupplier
from data.image_tuple_dataset import ImageTupleDataset

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for segmentation evaluation"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Intersection over Union) score"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def train_epoch(model, loader, loss_fn, optimizer, device, gradient_clipping=None):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for inputs, masks, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Gradient clipping
        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()

        # Calculate metrics
        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, labels)
        total_iou += iou_score(outputs, labels)
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for inputs, masks, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Calculate metrics
        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, labels)
        total_iou += iou_score(outputs, labels)
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


def main(cfg):
    """
    Train a U-Net with explicit parameters. Components take only arguments; config is loaded at invocation.
    """
    train_cfg = cfg["DLUnet"]

    setup_logging()
    logger = get_logger(__name__)

    device_name = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    data_tuples = DatasetSupplier.get_dataset(cfg["image_dir"])

    resize_shape = train_cfg.get("resize_shape", cfg.get("resize_shape", (256, 256)))

    val_split = train_cfg.get("val_split", 0.2)
    test_split = train_cfg.get("test_split", 0.15)
    random_seed = train_cfg.get("random_seed", 42)
    train_tuples, val_tuples, test_tuples = DatasetSupplier.train_val_test_split(
        data_tuples, val_split=val_split, test_split=test_split, seed=random_seed
    )

    train_image_dataset = ImageTupleDataset(
        train_tuples,
        image_size=tuple(resize_shape),
    )

    val_image_dataset = ImageTupleDataset(
        val_tuples,
        image_size=tuple(resize_shape),
    )

    test_image_dataset = ImageTupleDataset(
        test_tuples,
        image_size=tuple(resize_shape),
    )

    train_patch_dataset = PatchDataset(
        train_image_dataset,
        train_cfg["patch_size"], train_cfg["stride"],
        augment=train_cfg.get("use_augmentation", False)
    )
    val_patch_dataset = PatchDataset(
        val_image_dataset,
        train_cfg["patch_size"], train_cfg["stride"]
    )
    test_patch_dataset = PatchDataset(
        test_image_dataset,
        train_cfg["patch_size"], train_cfg["stride"]
    )

    train_size = len(train_patch_dataset)
    val_size = len(val_patch_dataset)
    test_size = len(test_patch_dataset)

    # Optionally use a WeightedRandomSampler to balance patches with and without vessels
    # Also compute labels list for pixel-wise statistics
    labels_list = [patch[2] for patch in train_patch_dataset.patches]
    balance_patches = train_cfg.get("balance_patches", True)
    if balance_patches:
        # compute simple per-patch label presence
        import numpy as _np
        labels = labels_list
        pos_flags = _np.array([_np.mean(l) > 0.0 for l in labels])
        pos_count = int(pos_flags.sum())
        neg_count = len(pos_flags) - pos_count
        logger.info(f"Train patch stats: {len(pos_flags)} patches ({pos_count} pos, {neg_count} neg)")
            # Also compute per-patch vessel pixel fraction and overall class ratio
            patch_pixel_fracs = _np.array([_np.mean(l) for l in labels])
            avg_pixel_frac = patch_pixel_fracs.mean() if len(patch_pixel_fracs) > 0 else 0.0
            logger.info(f"Average vessel fraction per patch: {avg_pixel_frac:.6f}")
        if pos_count == 0:
            logger.warning("No positive patches found — training may be invalid!")
        # Weight inversely proportional to class frequency
        weights = _np.where(pos_flags, 1.0 / (pos_count + 1e-6), 1.0 / (neg_count + 1e-6))
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_patch_dataset,
            batch_size=train_cfg["batch_size"],
            sampler=sampler,
            num_workers=train_cfg.get("num_workers", 0)
        )
    else:
        train_loader = DataLoader(
            train_patch_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0)
        )
    val_loader = DataLoader(
        val_patch_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0)
    )
    test_loader = DataLoader(
        test_patch_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0)
    )
    logger.info(f"Dataset split: {train_size} training patches, {val_size} validation patches, {test_size} test patches")

    # Compute dataset class imbalance and choose weight for CE/Focal
    # Use pixel-level ratio to set pos weighting
    try:
        total_pos_pixels = sum([l.sum() for l in labels_list])
        total_pixels = sum([l.size for l in labels_list])
        pos_pixel_ratio = float(total_pos_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        ce_weight = (1.0 - pos_pixel_ratio) / (pos_pixel_ratio + 1e-9) if pos_pixel_ratio > 0 else 1.0
        logger.info(f"Global pixel-level vessel fraction: {pos_pixel_ratio:.6f}; computed CE alpha/weight: {ce_weight:.3f}")
    except Exception:
        ce_weight = 1.0
        logger.warning("Could not compute pixel-level class ratio; using default CE/focal weight=1.0")

    # Model, loss, optimizer
    model = UNet(
        train_cfg["n_channels"],
        train_cfg["n_classes"],
        base_c=train_cfg.get("base_c", 64),
        dropout_rate=train_cfg.get("dropout_rate", 0.0)
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg.get("weight_decay", 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=train_cfg.get("scheduler_patience", 7),
        factor=train_cfg.get("scheduler_factor", 0.5),
        min_lr=train_cfg.get("min_lr", 1e-6)
    )

    loss_fn = DiceCEWithWeight(
        weight=ce_weight,
        device=device, 
        use_focal=train_cfg.get("use_focal_loss", True)    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
    gradient_clipping = train_cfg.get("gradient_clipping", 1.0)

    logger.info("Training configuration:")
    logger.info(f"   - Learning Rate: {train_cfg['learning_rate']}")
    logger.info(f"   - Weight Decay: {train_cfg.get('weight_decay', 0)}")
    logger.info(f"   - Early Stopping: {early_stopping_patience} epochs")
    logger.info(f"   - Scheduler Patience: {train_cfg.get('scheduler_patience', 5)} epochs")
    logger.info(f"   - Gradient Clipping: {gradient_clipping}")
    logger.info(f"   - Using Focal Loss: {train_cfg.get('use_focal_loss', True)}")
    logger.info("")

    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, gradient_clipping)
        val_metrics = eval_epoch(model, val_loader, loss_fn, device)

        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch}/{train_cfg['epochs']} — "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(Dice: {train_metrics['dice']:.3f}, IoU: {train_metrics['iou']:.3f}) — "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"(Dice: {val_metrics['dice']:.3f}, IoU: {val_metrics['iou']:.3f}) — "
            f"LR: {current_lr:.2e}")

        ckpt_path = os.path.join(checkpoint_dir, f"unet_epoch{epoch:03d}.pt")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': current_lr
        }, ckpt_path)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, "unet_best.pt")
            )
            logger.info(f"New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
            break

        if current_lr < train_cfg.get("min_lr", 1e-6):
            logger.info(f"\nStopping training: learning rate {current_lr:.2e} below minimum threshold")
            break

    logger.info("\n" + "-"*50)
    logger.info("Training completed. Evaluating on test set")
    logger.info("-"*50)
    best_model_path = os.path.join(checkpoint_dir, "unet_best.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from: {best_model_path}")

    test_metrics = eval_epoch(model, test_loader, loss_fn, device)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f} "
        f"(Dice: {test_metrics['dice']:.3f}, IoU: {test_metrics['iou']:.3f})")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")

    test_results = {
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }

    results_path = os.path.join(checkpoint_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a U-Net model")
    parser.add_argument("--config", default="config.yaml", help="Path to training config YAML")
    args = parser.parse_args()
    cfg = load_config_yaml(args.config)
    main(cfg)
