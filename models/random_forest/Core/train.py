import os
import sys
import json
import argparse
import logging
from pathlib import Path

import joblib

from .model import RandomForestModel
from .dataset import RandomForestFeatureDataset
from data.image_tuple_dataset import ImageTupleDataset
from data.dataset_supplier import DatasetSupplier
from util import load_config_yaml, calculate_comprehensive_metrics, print_metrics_summary, setup_logging, get_logger, set_all_seeds


def main(cfg):
    """Enhanced training pipeline with comprehensive error handling"""

    setup_logging()
    logger = get_logger(__name__)
    
    try:
        train_cfg = cfg["DLRandomForest"]
        logger.info("")
        logger.info("Random Forest training started")
        logger.info("-" * 60)
        
        set_all_seeds(train_cfg.get("random_state", 42))
        
        logger.info("Loading and preparing dataset...")
        data_tuples = DatasetSupplier.get_dataset(cfg["image_dir"])
        
        train_tuples, val_tuples, test_tuples = DatasetSupplier.train_val_test_split(
            data_tuples, 
            val_split=train_cfg.get("validation_size", 0.2),
            test_split=train_cfg.get("test_size", 0.15),
            seed=train_cfg.get("random_state", 42)
        )
        
        logger.info(f"Dataset split: {len(train_tuples)} train, {len(val_tuples)} val, {len(test_tuples)} test")
        
        resize_shape = train_cfg.get("resize_shape", [256, 256])
        
        train_image_dataset = ImageTupleDataset(
            train_tuples, 
            image_size=tuple(resize_shape),
        )
        
        val_image_dataset = ImageTupleDataset(
            val_tuples, 
            image_size=tuple(resize_shape),
        )
        
        logger.debug(f"Test tuples basenames: {[tuple_item[0] for tuple_item in test_tuples]}")

        test_image_dataset = ImageTupleDataset(
            test_tuples, 
            image_size=tuple(resize_shape),
        )
        
        logger.info("Creating feature datasets with processing...")
        
        train_rf_dataset = RandomForestFeatureDataset(
            image_tuple_dataset=train_image_dataset,
            patch_size=train_cfg.get("patch_size", 15),
            patches_per_class=train_cfg.get("patches_per_class", 75000),
            all_patches=train_cfg.get("all_patches", False),
            augment=True,
            balance=train_cfg.get("balance_train", True),
            #cache_dir=train_cfg.get("cache_dir"),
            n_workers=train_cfg.get("n_workers", 4),
            scaler=None,
            fit_scaler=True
        )
        joblib.dump(train_rf_dataset.scaler, "checkpoints/random_forest/scaler.pkl")
        
        val_rf_dataset = RandomForestFeatureDataset(
            image_tuple_dataset=val_image_dataset,
            patch_size=train_cfg.get("patch_size", 15),
            patches_per_class=train_cfg.get("patches_per_class", 75000),
            all_patches=train_cfg.get("all_patches", False),
            augment=False,  
            #cache_dir=train_cfg.get("cache_dir"),
            n_workers=train_cfg.get("n_workers", 4),
            scaler=train_rf_dataset.scaler, 
            fit_scaler=False 
        )
        
        test_rf_dataset = RandomForestFeatureDataset(
            image_tuple_dataset=test_image_dataset,
            patch_size=train_cfg.get("patch_size", 15),
            patches_per_class=train_cfg.get("patches_per_class", 75000),
            all_patches=train_cfg.get("all_patches", False),
            augment=False,
            #cache_dir=train_cfg.get("cache_dir"),
            n_workers=train_cfg.get("n_workers", 4),
            scaler=train_rf_dataset.scaler, 
            fit_scaler=False   
        )

        X_train, y_train = train_rf_dataset.get_all_data()
        X_val, y_val = val_rf_dataset.get_all_data()
        X_test, y_test = test_rf_dataset.get_all_data()
        
        logger.info(f"Feature matrix shapes:")
        logger.info(f"  - Train: {X_train.shape}")
        logger.info(f"  - Val: {X_val.shape}")
        logger.info(f"  - Test: {X_test.shape}")
        
        logger.info("Initializing and training Random Forest model...")
        model = RandomForestModel(train_cfg)

        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        model.train(X_train, y_train, feature_names)
        
        logger.info("Evaluating model performance...")

        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)
        train_metrics = calculate_comprehensive_metrics(y_train, train_pred, train_proba)
        print_metrics_summary(train_metrics, "Training")

        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        val_metrics = calculate_comprehensive_metrics(y_val, val_pred, val_proba)
        print_metrics_summary(val_metrics, "Validation")
        
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)
        test_metrics = calculate_comprehensive_metrics(y_test, test_pred, test_proba)
        print_metrics_summary(test_metrics, "Test")
        
        logger.info("Analyzing feature importance")
        importance_df = model.analyze_feature_importance(top_n=20)
        logger.info("Top 20 feature importances:")
        logger.info(importance_df.to_string(index=False))
        
        checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints/random_forest")
        model_name = train_cfg.get("model_name", "rf_model.pkl")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, model_name)
        model.save_model(model_path)

        if train_cfg.get("save_feature_importance", True):
            importance_path = os.path.join(checkpoint_dir, "feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")

        results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'config': train_cfg,
            'dataset_info': {
                'train_size': len(train_rf_dataset),
                'val_size': len(val_rf_dataset),
                'test_size': len(test_rf_dataset),
                'n_features': X_train.shape[1],
                'feature_names': feature_names
            }
        }
        
        results_path = os.path.join(checkpoint_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("")
        logger.info("Results saved:")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Results: {results_path}")

        logger.info("")
        logger.info("Training summary")
        logger.info("-" * 60)
        logger.info(f"Test Accuracy:     {test_metrics['accuracy']:.3f}")
        logger.info(f"Test F1-Score:     {test_metrics['f1']:.3f}")
        logger.info(f"Test Dice:         {test_metrics['dice']:.3f}")
        logger.info(f"Test Sensitivity:  {test_metrics['sensitivity']:.3f}")
        logger.info(f"Test Specificity:  {test_metrics['specificity']:.3f}")
        logger.info(f"Test AUC:          {test_metrics['auc']:.3f}")
        logger.info("-" * 60)
        
        return model, results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Enhanced Random Forest model")
    parser.add_argument("--config", default="config.yaml", 
                       help="Path to training config YAML")
    args = parser.parse_args()
    
    cfg = load_config_yaml(args.config)
    model, results = main(cfg)
