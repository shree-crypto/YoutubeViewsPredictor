"""
YouTube Views Predictor - Utilities Package

This package contains modules for feature engineering and model training.
"""

from .feature_engineering import FeatureExtractor, get_optimal_features
from .model_training import YouTubeViewsPredictor, create_sample_dataset

__all__ = [
    'FeatureExtractor',
    'get_optimal_features',
    'YouTubeViewsPredictor',
    'create_sample_dataset'
]
