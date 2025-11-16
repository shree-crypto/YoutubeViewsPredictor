"""
Comprehensive unit tests for YouTube Views Predictor

Run with: pytest test_comprehensive.py -v --cov=utils --cov-report=html
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile
import shutil

from utils.feature_engineering import FeatureExtractor, get_optimal_features
from utils.model_training import YouTubeViewsPredictor, create_sample_dataset
from utils.config import Config, get_config
from utils.validators import (
    VideoMetadata,
    PredictionRequest,
    validate_video_data,
    validate_prediction_request
)


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create FeatureExtractor instance."""
        return FeatureExtractor()
    
    def test_extract_title_features_normal(self, extractor):
        """Test title feature extraction with normal input."""
        features = extractor.extract_title_features("Amazing Tutorial! Learn Python in 10 Minutes")
        
        assert features['title_length'] == 47
        assert features['title_word_count'] == 7
        assert features['has_exclamation'] == 1
        assert features['has_question_mark'] == 0
        assert features['title_digit_count'] == 2  # "10"
        assert isinstance(features['title_sentiment_polarity'], float)
    
    def test_extract_title_features_empty(self, extractor):
        """Test title feature extraction with empty string."""
        features = extractor.extract_title_features("")
        
        assert features['title_length'] == 0
        assert features['title_word_count'] == 0
        assert features['has_exclamation'] == 0
    
    def test_extract_title_features_none(self, extractor):
        """Test title feature extraction with None."""
        features = extractor.extract_title_features(None)
        
        assert features['title_length'] == 0
        assert features['title_word_count'] == 0
    
    def test_extract_temporal_features(self, extractor):
        """Test temporal feature extraction."""
        dt = datetime(2024, 1, 19, 19, 30, 0)  # Friday 7:30 PM
        features = extractor.extract_temporal_features(dt)
        
        assert features['publish_hour'] == 19
        assert features['publish_day_of_week'] == 4  # Friday (0=Monday)
        assert features['publish_month'] == 1
        assert features['is_weekend'] == 0  # Friday is not weekend (Sat/Sun)
        assert features['is_peak_hour'] == 1  # 7 PM is peak
    
    def test_extract_temporal_features_weekend(self, extractor):
        """Test temporal feature extraction for weekend."""
        dt = datetime(2024, 1, 20, 15, 0, 0)  # Saturday 3 PM
        features = extractor.extract_temporal_features(dt)
        
        assert features['publish_day_of_week'] == 5  # Saturday
        assert features['is_weekend'] == 1
    
    def test_extract_duration_features(self, extractor):
        """Test duration feature extraction."""
        features = extractor.extract_duration_features(600)  # 10 minutes
        
        assert features['duration_seconds'] == 600
        assert features['duration_minutes'] == 10.0
        assert features['is_short_video'] == 0
        assert features['is_medium_video'] == 1
        assert features['is_long_video'] == 1  # >= 10 min
    
    def test_extract_duration_features_short(self, extractor):
        """Test duration feature extraction for short video."""
        features = extractor.extract_duration_features(45)  # 45 seconds
        
        assert features['is_short_video'] == 1
        assert features['is_medium_video'] == 0
        assert features['is_long_video'] == 0
    
    def test_extract_tags_features(self, extractor):
        """Test tags feature extraction."""
        features = extractor.extract_tags_features("python,tutorial,coding,programming")
        
        assert features['tags_count'] == 4
        assert features['avg_tag_length'] > 0
    
    def test_extract_tags_features_empty(self, extractor):
        """Test tags feature extraction with empty string."""
        features = extractor.extract_tags_features("")
        
        assert features['tags_count'] == 0
        assert features['avg_tag_length'] == 0
    
    def test_extract_description_features(self, extractor):
        """Test description feature extraction."""
        desc = "Learn Python! Visit https://python.org and https://github.com"
        features = extractor.extract_description_features(desc)
        
        assert features['description_length'] == len(desc)
        assert features['description_word_count'] == 7
        assert features['description_link_count'] == 2
    
    def test_extract_all_features(self, extractor):
        """Test extraction of all features together."""
        data = {
            'title': 'Python Tutorial 2024!',
            'duration': 600,
            'tags': 'python,tutorial',
            'publish_time': datetime(2024, 1, 15, 18, 0, 0),
            'description': 'Learn Python from scratch.'
        }
        
        features = extractor.extract_all_features(data)
        
        # Check all feature groups are present
        assert 'title_length' in features
        assert 'publish_hour' in features
        assert 'duration_seconds' in features
        assert 'tags_count' in features
        assert 'description_length' in features
        
        # Verify some values
        assert features['publish_hour'] == 18
        assert features['is_peak_hour'] == 1


class TestYouTubeViewsPredictor:
    """Test cases for YouTubeViewsPredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        df = create_sample_dataset(n_samples=100, output_path='/tmp/test_data.csv')
        feature_cols = [col for col in df.columns if col != 'views']
        X = df[feature_cols]
        y = df['views']
        return X, y
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        assert predictor.model_type == 'xgboost'
        assert predictor.model is None
        assert predictor.feature_names is None
    
    def test_predictor_invalid_model_type(self):
        """Test predictor with invalid model type."""
        with pytest.raises(ValueError):
            YouTubeViewsPredictor(model_type='invalid')
    
    def test_train_model(self, sample_data):
        """Test model training."""
        X, y = sample_data
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        results = predictor.train(X, y, test_size=0.2, random_state=42)
        
        assert 'train_metrics' in results
        assert 'test_metrics' in results
        assert 'mae' in results['test_metrics']
        assert 'rmse' in results['test_metrics']
        assert 'r2' in results['test_metrics']
        assert predictor.model is not None
        assert predictor.feature_names is not None
    
    def test_train_model_empty_data(self):
        """Test training with empty data."""
        predictor = YouTubeViewsPredictor()
        X = pd.DataFrame()
        y = pd.Series()
        
        with pytest.raises(ValueError):
            predictor.train(X, y)
    
    def test_train_model_mismatched_lengths(self, sample_data):
        """Test training with mismatched X and y lengths."""
        X, y = sample_data
        predictor = YouTubeViewsPredictor()
        
        with pytest.raises(ValueError):
            predictor.train(X, y[:10])  # y is shorter
    
    def test_predict(self, sample_data):
        """Test making predictions."""
        X, y = sample_data
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        predictor.train(X, y, test_size=0.2, random_state=42)
        
        # Make prediction on first sample
        predictions = predictor.predict(X.iloc[0:1])
        
        assert len(predictions) == 1
        assert predictions[0] >= 0
    
    def test_predict_without_training(self):
        """Test prediction without training."""
        predictor = YouTubeViewsPredictor()
        X = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
        
        with pytest.raises(ValueError):
            predictor.predict(X)
    
    def test_predict_empty_data(self, sample_data):
        """Test prediction with empty data."""
        X, y = sample_data
        predictor = YouTubeViewsPredictor()
        predictor.train(X, y, test_size=0.2, random_state=42)
        
        with pytest.raises(ValueError):
            predictor.predict(pd.DataFrame())
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model."""
        X, y = sample_data
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Train and save
            predictor = YouTubeViewsPredictor(model_type='xgboost')
            predictor.train(X, y, test_size=0.2, random_state=42)
            predictor.save_model(temp_dir)
            
            # Load and predict
            predictor2 = YouTubeViewsPredictor(model_type='xgboost')
            predictor2.load_model(temp_dir)
            
            # Compare predictions
            pred1 = predictor.predict(X.iloc[0:1])
            pred2 = predictor2.predict(X.iloc[0:1])
            
            np.testing.assert_array_almost_equal(pred1, pred2)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_top_features(self, sample_data):
        """Test getting top features."""
        X, y = sample_data
        predictor = YouTubeViewsPredictor()
        predictor.train(X, y, test_size=0.2, random_state=42)
        
        top_features = predictor.get_top_features(n=5)
        
        assert top_features is not None
        assert len(top_features) == 5
        assert all(isinstance(f[0], str) and isinstance(f[1], float) for f in top_features)


class TestValidators:
    """Test cases for data validators."""
    
    def test_video_metadata_valid(self):
        """Test valid video metadata."""
        data = {
            'title': 'Test Video',
            'description': 'Test description',
            'tags': 'test,video',
            'duration': 600,
            'publish_time': datetime.now()
        }
        
        metadata = VideoMetadata(**data)
        assert metadata.title == 'Test Video'
        assert metadata.duration == 600
    
    def test_video_metadata_empty_title(self):
        """Test video metadata with empty title."""
        data = {
            'title': '',
            'duration': 600,
            'publish_time': datetime.now()
        }
        
        with pytest.raises(ValueError):
            VideoMetadata(**data)
    
    def test_video_metadata_negative_duration(self):
        """Test video metadata with negative duration."""
        data = {
            'title': 'Test',
            'duration': -10,
            'publish_time': datetime.now()
        }
        
        with pytest.raises(ValueError):
            VideoMetadata(**data)
    
    def test_prediction_request_valid(self):
        """Test valid prediction request."""
        data = {
            'title': 'Test Video',
            'duration_minutes': 10.0,
            'publish_hour': 18,
            'publish_date': '2024-01-15'
        }
        
        request = PredictionRequest(**data)
        assert request.title == 'Test Video'
        assert request.publish_hour == 18
    
    def test_prediction_request_invalid_date(self):
        """Test prediction request with invalid date format."""
        data = {
            'title': 'Test',
            'duration_minutes': 10.0,
            'publish_hour': 18,
            'publish_date': 'invalid-date'
        }
        
        with pytest.raises(ValueError):
            PredictionRequest(**data)


class TestConfig:
    """Test cases for configuration management."""
    
    def test_config_singleton(self):
        """Test that Config is a singleton."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2
    
    def test_get_config(self):
        """Test getting configuration values."""
        config = get_config()
        
        # Test getting nested values
        model_type = config.get('model.type')
        assert model_type is not None
    
    def test_get_config_default(self):
        """Test getting config with default value."""
        config = get_config()
        value = config.get('nonexistent.key', 'default_value')
        assert value == 'default_value'
    
    def test_set_config(self):
        """Test setting configuration values."""
        config = get_config()
        config.set('test.key', 'test_value')
        assert config.get('test.key') == 'test_value'


class TestCreateSampleDataset:
    """Test cases for sample dataset creation."""
    
    def test_create_sample_dataset(self):
        """Test creating sample dataset."""
        df = create_sample_dataset(n_samples=50, output_path='/tmp/test_sample.csv')
        
        assert len(df) == 50
        assert 'views' in df.columns
        assert df['views'].min() >= 0
        assert os.path.exists('/tmp/test_sample.csv')
    
    def test_create_sample_dataset_invalid_size(self):
        """Test creating sample dataset with invalid size."""
        with pytest.raises(ValueError):
            create_sample_dataset(n_samples=-10)


class TestOptimalFeatures:
    """Test cases for optimal features recommendations."""
    
    def test_get_optimal_features(self):
        """Test getting optimal features recommendations."""
        recommendations = get_optimal_features()
        
        assert 'title_recommendations' in recommendations
        assert 'temporal_recommendations' in recommendations
        assert 'duration_recommendations' in recommendations
        assert 'metadata_recommendations' in recommendations
        
        assert len(recommendations['title_recommendations']) > 0
        assert all(isinstance(r, str) for r in recommendations['title_recommendations'])


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_complete_pipeline(self):
        """Test complete prediction pipeline."""
        # 1. Create feature extractor
        extractor = FeatureExtractor()
        
        # 2. Create sample video data
        video_data = {
            'title': 'Complete Python Tutorial for Beginners 2024!',
            'duration': 720,
            'tags': 'python,tutorial,programming,coding,beginners',
            'publish_time': datetime(2024, 1, 19, 19, 0, 0),
            'description': 'Learn Python from scratch. Perfect for beginners!'
        }
        
        # 3. Extract features
        features = extractor.extract_all_features(video_data)
        features_df = pd.DataFrame([features])
        
        # 4. Create and train model
        sample_df = create_sample_dataset(n_samples=100, output_path='/tmp/test_e2e.csv')
        feature_cols = [col for col in sample_df.columns if col != 'views']
        X_train = sample_df[feature_cols]
        y_train = sample_df['views']
        
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        predictor.train(X_train, y_train, test_size=0.2, random_state=42)
        
        # 5. Make prediction
        prediction = predictor.predict(features_df)
        
        assert len(prediction) == 1
        assert prediction[0] >= 0
        
        # 6. Get top features
        top_features = predictor.get_top_features(n=5)
        assert len(top_features) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=utils', '--cov-report=term-missing'])
