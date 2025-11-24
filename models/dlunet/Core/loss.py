import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in segmentation"""
    def __init__(self, alpha=1, gamma=2, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class DiceCEWithWeight(nn.Module):
    def __init__(self, weight=0.9, device=None, use_focal=True):
        super().__init__()
        self.dice = DiceLoss()
        self.use_focal = use_focal
        
        if use_focal:
            self.ce = FocalLoss(alpha=weight, gamma=2)
        else:
            pos_weight = torch.tensor([weight])
            if device is not None:
                pos_weight = pos_weight.to(device)
            self.ce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        if self.use_focal:
            ce_loss = self.ce(pred, target)
        else:
            ce_loss = self.ce(pred, target)
        return 0.5 * dice_loss + 0.5 * ce_loss
