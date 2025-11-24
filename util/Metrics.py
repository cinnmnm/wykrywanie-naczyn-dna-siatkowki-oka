import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from typing import Dict, Optional
from .logging_config import get_logger

logger = get_logger(__name__)

def calculate_comprehensive_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray, 
                                  y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate comprehensive metrics for medical image segmentation [1][2]"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    metrics['dice'] = metrics['f1']
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp + 1e-8)
    
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.0
    
    metrics['sensitivity'] = metrics['recall']
    
    return metrics

def print_metrics_summary(metrics: Dict[str, float], dataset_name: str = ""):
    """Log formatted metrics summary"""
    logger.info(f"{dataset_name} metrics:")
    logger.info("-" * 40)
    for metric, value in metrics.items():
        logger.info(f"  {metric.capitalize():<12}: {value:.4f}")
    logger.info("-" * 40)
