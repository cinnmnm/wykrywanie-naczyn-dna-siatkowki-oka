import random
import numpy as np
import os
from .logging_config import get_logger

logger = get_logger(__name__)

def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility [1]"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import cv2
        cv2.setRNGSeed(seed)
    except ImportError:
        pass
    
    logger.info(f"All random seeds set to {seed}")