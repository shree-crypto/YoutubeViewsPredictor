"""
YouTube Views Predictor - Utilities Package

This package contains modules for feature engineering, model training,
configuration management, logging, and validation.
"""

from .feature_engineering import FeatureExtractor, get_optimal_features
from .model_training import YouTubeViewsPredictor, create_sample_dataset
from .config import Config, get_config
from .logging_config import setup_logging, get_logger
from .validators import (
    VideoMetadata,
    PredictionRequest,
    PredictionResponse,
    ModelMetrics,
    validate_video_data,
    validate_prediction_request
)

__all__ = [
    'FeatureExtractor',
    'get_optimal_features',
    'YouTubeViewsPredictor',
    'create_sample_dataset',
    'Config',
    'get_config',
    'setup_logging',
    'get_logger',
    'VideoMetadata',
    'PredictionRequest',
    'PredictionResponse',
    'ModelMetrics',
    'validate_video_data',
    'validate_prediction_request'
]
