import numpy as np
from sklearn.metrics import confusion_matrix
import yaml
import os
from .logging_config import get_logger
logger = get_logger(__name__)

class Evaluate:
    @staticmethod
    def confusion(y_true, y_pred, mask=None):
        """
        Return confusion matrix (TN, FP, FN, TP) for binary classification.
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        y_true_flat = np.where(y_true_flat > 1, 1, y_true_flat)
        y_pred_flat = np.where(y_pred_flat > 1, 1, y_pred_flat)

        if mask is not None:
            mask_flat = mask.flatten() != 0
            if mask_flat.size != y_true_flat.size:
                raise ValueError(f"Mask shape incompatible with y_true/y_pred shape after flattening. y_true_flat.size: {y_true_flat.size}, mask_flat.size: {mask_flat.size}")
            y_true_masked = y_true_flat[mask_flat]
            y_pred_masked = y_pred_flat[mask_flat]
        else:
            y_true_masked = y_true_flat
            y_pred_masked = y_pred_flat
        
        if y_true_masked.size == 0:
            return {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}

        cm_values = confusion_matrix(y_true_masked, y_pred_masked, labels=[0,1]).ravel()
        tn, fp, fn, tp = cm_values

        return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

    @staticmethod
    def accuracy(y_true, y_pred, mask=None):
        """
        Computes accuracy using the confusion matrix.
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        tp = cm['TP']
        tn = cm['TN']
        fp = cm['FP']
        fn = cm['FN']
        
        total = tp + tn + fp + fn
        if total == 0:
            return 0.0 
        
        return (tp + tn) / total

    @staticmethod
    def sensitivity(y_true, y_pred, mask=None):
        """
        Computes sensitivity (True Positive Rate or Recall).
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        tp = cm['TP']
        fn = cm['FN']
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def specificity(y_true, y_pred, mask=None):
        """
        Computes specificity (True Negative Rate).
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        tn = cm['TN']
        fp = cm['FP']
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def print_confusion_matrix(y_true, y_pred, mask=None):
        """
        Calls the confusion method and prints the confusion matrix in a readable format.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        logger.info("Confusion Matrix:")
        logger.info(f"TN: {cm['TN']}  FP: {cm['FP']}")
        logger.info(f"FN: {cm['FN']}  TP: {cm['TP']}")

