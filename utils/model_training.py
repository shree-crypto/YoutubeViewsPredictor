"""
Model Training Module for YouTube Views Prediction

Implements ensemble learning approach combining:
- XGBoost for gradient boosting
- Feature importance analysis
- Cross-validation
- Model persistence
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import json
import os
from typing import Dict, Tuple, Optional, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeViewsPredictor:
    """Main predictor class for YouTube views."""
    
    def __init__(self, model_type: str = 'xgboost') -> None:
        """
        Initialize predictor.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in ['xgboost', 'lightgbm']:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'xgboost' or 'lightgbm'")
        
        self.model_type = model_type
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        
        logger.info(f"Initialized YouTubeViewsPredictor with model_type={model_type}")
        
    def prepare_features(self, X):
        """Prepare and scale features."""
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        return self.scaler.fit_transform(X)
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
              random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature dataframe
            y: Target variable (view count)
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            dict: Training metrics
            
        Raises:
            ValueError: If input data is invalid
        """
        if X.empty or y.empty:
            raise ValueError("Input data cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        logger.info(f"Starting model training with {len(X)} samples")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        try:
            if self.model_type == 'xgboost':
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif self.model_type == 'lightgbm':
                self.model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Fit model
            logger.info("Fitting model...")
            self.model.fit(X_train_scaled, y_train)
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'mae': float(mean_absolute_error(y_train, y_train_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'r2': float(r2_score(y_train, y_train_pred))
        }
        
        test_metrics = {
            'mae': float(mean_absolute_error(y_test, y_test_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'r2': float(r2_score(y_test, y_test_pred))
        }
        
        logger.info(f"Training metrics - MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.4f}")
        logger.info(f"Test metrics - MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names,
                [float(x) for x in self.model.feature_importances_]
            ))
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature dataframe or dict
            
        Returns:
            Predicted view count
            
        Raises:
            ValueError: If model not trained or input invalid
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Handle single prediction (dict input)
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        if X.empty:
            raise ValueError("Input data cannot be empty")
        
        # Ensure correct feature order
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            X = X[self.feature_names]
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            predictions = self.model.predict(X_scaled)
            
            # Ensure non-negative predictions
            predictions = np.maximum(predictions, 0)
            
            logger.debug(f"Made {len(predictions)} predictions")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_model(self, model_dir: str = 'models') -> None:
        """
        Save model and scaler to disk.
        
        Args:
            model_dir: Directory to save model files
            
        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'{self.model_type}_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            config_path = os.path.join(model_dir, 'model_config.json')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            config = {
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Model saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_dir: str = 'models') -> None:
        """
        Load model and scaler from disk.
        
        Args:
            model_dir: Directory containing model files
            
        Raises:
            FileNotFoundError: If model files not found
        """
        model_path = os.path.join(model_dir, f'{self.model_type}_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        config_path = os.path.join(model_dir, 'model_config.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.feature_names = config.get('feature_names')
                self.feature_importance = config.get('feature_importance')
            
            logger.info(f"Model loaded from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_top_features(self, n: int = 10) -> Optional[List[Tuple[str, float]]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples or None if not available
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available")
            return None
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:n]


def create_sample_dataset(n_samples: int = 1000, 
                         output_path: str = 'data/processed/sample_data.csv') -> pd.DataFrame:
    """
    Create a synthetic sample dataset for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the dataset
        
    Returns:
        DataFrame with synthetic data
        
    Raises:
        ValueError: If n_samples is invalid
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    logger.info(f"Creating sample dataset with {n_samples} samples")
    
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'title_length': np.random.randint(20, 100, n_samples),
        'title_word_count': np.random.randint(3, 15, n_samples),
        'title_uppercase_ratio': np.random.uniform(0, 0.3, n_samples),
        'title_digit_count': np.random.randint(0, 5, n_samples),
        'title_special_char_count': np.random.randint(0, 10, n_samples),
        'title_sentiment_polarity': np.random.uniform(-0.5, 0.5, n_samples),
        'title_sentiment_subjectivity': np.random.uniform(0, 1, n_samples),
        'has_question_mark': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'has_exclamation': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'title_capitalized_words': np.random.randint(0, 3, n_samples),
        'publish_hour': np.random.randint(0, 24, n_samples),
        'publish_day_of_week': np.random.randint(0, 7, n_samples),
        'publish_month': np.random.randint(1, 13, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'is_peak_hour': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'duration_seconds': np.random.randint(60, 3600, n_samples),
        'duration_minutes': np.random.uniform(1, 60, n_samples),
        'is_short_video': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'is_medium_video': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'is_long_video': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'tags_count': np.random.randint(0, 20, n_samples),
        'avg_tag_length': np.random.uniform(5, 20, n_samples),
        'description_length': np.random.randint(0, 1000, n_samples),
        'description_word_count': np.random.randint(0, 200, n_samples),
        'description_link_count': np.random.randint(0, 5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic target (views) with some logic
    # More views for: optimal title length, peak hours, weekend, more tags
    views = (
        10000 +
        (100 - abs(df['title_length'] - 60)) * 100 +  # Optimal title length around 60
        df['is_peak_hour'] * 50000 +
        df['is_weekend'] * 30000 +
        df['tags_count'] * 1000 +
        df['has_question_mark'] * 20000 +
        df['title_sentiment_polarity'] * 10000 +
        (10 - abs(df['duration_minutes'] - 10)) * 1000 +  # Optimal duration around 10 min
        np.random.normal(0, 20000, n_samples)  # Add noise
    )
    
    df['views'] = np.maximum(views, 100).astype(int)  # Ensure positive views
    
    # Save dataset
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Sample dataset saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        raise
    
    return df
