import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
from util import get_logger
from typing import Dict, Any, List, Optional
import pandas as pd

class RandomForestModel:
    """
    Enhanced Random Forest model for retinal vessel segmentation [15].
    Includes comprehensive error handling and feature analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names_ = None
        
        self.logger = get_logger(__name__)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Random Forest with robust parameter handling"""
        try:           
            self.model = RandomForestClassifier(
                n_estimators=self.config.get("n_estimators", 400),
                max_depth=self.config.get("max_depth", 25),
                min_samples_split=self.config.get("min_samples_split", 3),
                min_samples_leaf=self.config.get("min_samples_leaf", 1),
                max_features=self.config.get("max_features", "sqrt"),
                bootstrap=self.config.get("bootstrap", True),
                n_jobs=self.config.get("n_jobs", -1),
                random_state=self.config.get("random_state", 42),
                class_weight=self.config.get("class_weight", "balanced"),
                criterion=self.config.get("criterion", "entropy"),
                max_leaf_nodes=self.config.get("max_leaf_nodes", 10000),
                min_impurity_decrease=self.config.get("min_impurity_decrease", 0.0005),
                oob_score=self.config.get("oob_score", True),
                verbose=self.config.get("verbose", 0)
            )
            
            self.logger.info("Random Forest model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Train the model with comprehensive logging and validation"""
        try:
            self.logger.info("Starting Random Forest training...")
            self.logger.info(f"Training samples: {X.shape[0]}")
            self.logger.info(f"Features: {X.shape[1]}")
            
            if len(np.unique(y)) < 2:
                raise ValueError("Training data must contain both classes")
            
            positive_ratio = np.mean(y)
            self.logger.info(f"Positive samples: {np.sum(y)} ({positive_ratio*100:.1f}%)")
            
            if feature_names:
                self.feature_names_ = feature_names
            else:
                self.feature_names_ = [f"Feature_{i}" for i in range(X.shape[1])]
            
            self.model.fit(X, y)
            self.is_trained = True
            
            if hasattr(self.model, 'oob_score_'):
                self.logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
            
            self.logger.info("Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with error handling"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities with error handling"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance with enhanced analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.model.feature_importances_
    
    def analyze_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze and return top feature importances as DataFrame"""
        if not self.is_trained:
            raise ValueError("Model must be trained for feature analysis")
        
        importances = self.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save model with error handling"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'config': self.config,
                'feature_names': self.feature_names_,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load model with error handling"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.config = model_data.get('config', {})
            self.feature_names_ = model_data.get('feature_names', None)
            self.is_trained = model_data.get('is_trained', False)
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
