from util import get_logger, setup_logging, load_config_yaml
from data.dataset_supplier import DatasetSupplier
from data.image_tuple_dataset import ImageTupleDataset
import numpy as np
from .model import RandomForestModel
from .dataset import RandomForestFeatureDataset
from sklearn.metrics import f1_score, accuracy_score

logger = get_logger(__name__)


def objective(params, X_train, y_train, X_val, y_val, train_cfg):
    """Objective function for hyperparameter optimization"""
    from hyperopt import STATUS_OK
    try:
        model_config = {
            **train_cfg,  
            **params     
        }
        
        rf_model = RandomForestModel(model_config)
        rf_model.train(X_train, y_train)
        
        logger.debug(f"Model config: {model_config}")

        y_pred = rf_model.predict(X_val)
        accuracy = np.mean(y_pred == y_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        intersection = np.sum((y_pred == 1) & (y_val == 1))
        dice = 2 * intersection / (np.sum(y_pred == 1) + np.sum(y_val == 1)) if (np.sum(y_pred == 1) + np.sum(y_val == 1)) > 0 else 0

        logger.info(f"Params: {params}")
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Dice: {dice:.4f}")

        return {'loss': -accuracy, 'status': STATUS_OK}
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization error: {e}")
        return {'loss': 0, 'status': STATUS_OK}
    except Exception as e:
        logger.error(f"Hyperparameter optimization error: {e}")
        return {'loss': 0, 'status': STATUS_OK}


def main(cfg_path: str = "config.yaml"):
    """Runs hyperparameter optimization for a RandomForest model using hyperopt.

    This function is safe to import (no heavy work at import time). Call `main()` or
    use `from models.random_forest.Core.hyperopt import main as hyperopt` to run.
    """
    setup_logging()
    cfg = load_config_yaml(cfg_path)
    train_cfg = cfg["DLRandomForest"]

    data_tuples = DatasetSupplier.get_dataset(cfg["image_dir"])    
    train_tuples, val_tuples, test_tuples = DatasetSupplier.train_val_test_split(
        data_tuples,
        val_split=train_cfg.get("validation_size", 0.2),
        test_split=train_cfg.get("test_size", 0.15),
        seed=train_cfg.get("random_state", 42)
    )

    logger.info(f"Dataset split: {len(train_tuples)} train, {len(val_tuples)} val, {len(test_tuples)} test")

    resize_shape = train_cfg.get("resize_shape", [256, 256])

    train_image_dataset = ImageTupleDataset(train_tuples, image_size=tuple(resize_shape))
    val_image_dataset = ImageTupleDataset(val_tuples, image_size=tuple(resize_shape))
    test_image_dataset = ImageTupleDataset(test_tuples, image_size=tuple(resize_shape))

    from hyperopt import hp

    space = {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500]),
        'max_depth': hp.choice('max_depth', [10, 15, 20, 25, 30, None]),
        'min_samples_split': hp.choice('min_samples_split', [2, 3, 5, 10]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [2, 3, 5, 8]),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 0.01),
        'max_leaf_nodes': hp.choice('max_leaf_nodes', [1000, 5000, 10000, None])
    }

    train_rf_dataset = RandomForestFeatureDataset(
        image_tuple_dataset=train_image_dataset,
        patch_size=train_cfg.get("patch_size", 15),
        patches_per_class=train_cfg.get("patches_per_class", 75000),
        all_patches=train_cfg.get("all_patches", False),
        augment=True,
        balance=train_cfg.get("balance_train", True),
        n_workers=train_cfg.get("n_workers", 4),
        scaler=None,
        fit_scaler=True
    )

    val_rf_dataset = RandomForestFeatureDataset(
        image_tuple_dataset=val_image_dataset,
        patch_size=train_cfg.get("patch_size", 15),
        patches_per_class=train_cfg.get("patches_per_class", 75000),
        all_patches=train_cfg.get("all_patches", False),
        augment=False,  
        n_workers=train_cfg.get("n_workers", 4),
        scaler=train_rf_dataset.scaler, 
        fit_scaler=False 
    )

    X_train, y_train = train_rf_dataset.get_all_data()
    X_val, y_val = val_rf_dataset.get_all_data()

    # Trials will be created inside main where hyperopt is imported to avoid importing
    # heavy third-party libraries at module import-time.
    best = None
    try:
        from hyperopt import fmin, tpe, hp, Trials
        trials = Trials()
        best = fmin(
            fn=lambda params: objective(params, X_train, y_train, X_val, y_val, train_cfg),
            space=space,
            algo=tpe.suggest,
            max_evals=train_cfg.get("hyperopt_max_evals", 10),
            trials=trials
        )
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
    finally:
        logger.info(f"Best hyperparameters: {best}")
        if best is None:
            return
        final_config = {**train_cfg, **best}
        final_model = RandomForestModel(final_config)
        final_model.train(X_train, y_train)
        model_save_path = train_cfg.get("model_save_path", "models/optimized_rf_model.pkl")
        final_model.save_model(model_save_path)


if __name__ == "__main__":
    main()
