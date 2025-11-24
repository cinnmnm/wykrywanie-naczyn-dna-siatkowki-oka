from sympy import Tuple
import torch
from models.dlunet import InferencePatchDataset, run_inference, UNet
from data.dataset_supplier import DatasetSupplier
from data.image_tuple_dataset import ImageTupleDataset
from filter.filter_segmentation import FilterSegmentation
import numpy as np
import cv2
import pickle
import os
from data.preprocessing import ImagePreprocessing
from torch.utils.data import DataLoader
from models.random_forest import RandomForestInference
from util import load_config_yaml, setup_logging, get_logger

try:
    setup_logging()
except Exception:
    pass

logger = get_logger(__name__)

class Controller:
    def run_filter(self, image):
        return FilterSegmentation.run(image)

    def run_ml(self, img_path, mask_path) -> np.ndarray:
        img_path = img_path.replace("\\", "/", 1)

        config = load_config_yaml('config.yaml')
        model_path = os.path.join(config['DLRandomForest']['checkpoint_dir'], config['DLRandomForest']['model_name'])

        try:
            inference_engine = RandomForestInference(model_path=model_path, config=config)
        except Exception as e:
            logger.exception("Error initializing inference engine: %s", e)
            raise

        try:
            pred_map, seg_map = inference_engine.predict_image(config['DLRandomForest'],
                image_path=img_path,
                mask_path=mask_path
            )
        except Exception as e:
            logger.exception("Error during batch prediction: %s", e)
            raise

        return seg_map

    def run_dl(self, img_path, mask_path):
        config = load_config_yaml('config.yaml')
        dl_config = config.get('DLUnet', {})
        
        device_name = dl_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)

        model_path = os.path.join(dl_config.get('checkpoint_dir', 'DLUnet/checkpoints'), 
                     dl_config.get('model_name', 'unet_best.pt'))
        
        model = UNet(
            dl_config["n_channels"],
            dl_config["n_classes"],
            base_c=dl_config.get("base_c", 64),
            dropout_rate=dl_config.get("dropout_rate", 0.0)
        )
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        resize_shape = dl_config.get("resize_shape", config.get("resize_shape", (256, 256)))
        data_tuples = [('inference', img_path, mask_path, mask_path)]

        image_dataset = ImageTupleDataset(data_tuples, resize_shape)
        
        inference_dataset = InferencePatchDataset(
            image_dataset,
            dl_config["patch_size"], 
            dl_config["stride"],
            resize_shape  
        )

        predictions = run_inference(model, inference_dataset, device, 
                      batch_size=dl_config.get("batch_size", 16))

        return predictions[0]
    
    def dummy(self, image):
        image = ImagePreprocessing.resize_and_normalize(image)
        image = (image > 0).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
