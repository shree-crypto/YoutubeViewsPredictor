"""
Simple tests for YouTube Views Predictor

Run with: python -m pytest test_basic.py
Or simply: python test_basic.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_engineering import FeatureExtractor, get_optimal_features
from utils.model_training import YouTubeViewsPredictor, create_sample_dataset
import pandas as pd
from datetime import datetime


def test_feature_extractor_title():
    """Test title feature extraction."""
    extractor = FeatureExtractor()
    
    # Test with normal title
    features = extractor.extract_title_features("How to Code in Python?")
    
    assert 'title_length' in features
    assert 'title_word_count' in features
    assert features['has_question_mark'] == 1
    assert features['title_word_count'] == 5
    print("✓ Title feature extraction works")


def test_feature_extractor_temporal():
    """Test temporal feature extraction."""
    extractor = FeatureExtractor()
    
    # Test with specific datetime
    dt = datetime(2024, 1, 15, 19, 0, 0)  # 7 PM on a Monday
    features = extractor.extract_temporal_features(dt)
    
    assert features['publish_hour'] == 19
    assert features['publish_day_of_week'] == 0  # Monday
    assert features['is_peak_hour'] == 1  # 7 PM is peak
    print("✓ Temporal feature extraction works")


def test_feature_extractor_duration():
    """Test duration feature extraction."""
    extractor = FeatureExtractor()
    
    # Test with 10 minutes (600 seconds)
    features = extractor.extract_duration_features(600)
    
    assert features['duration_seconds'] == 600
    assert features['duration_minutes'] == 10.0
    assert features['is_long_video'] == 1  # >= 10 minutes is long
    print("✓ Duration feature extraction works")


def test_feature_extractor_all():
    """Test extraction of all features."""
    extractor = FeatureExtractor()
    
    data = {
        'title': 'Amazing Tutorial!',
        'duration': 600,
        'tags': 'tutorial,learning,howto',
        'publish_time': '2024-01-15 18:00:00',
        'description': 'Learn something new in this tutorial.'
    }
    
    features = extractor.extract_all_features(data)
    
    # Should have all feature types
    assert 'title_length' in features
    assert 'publish_hour' in features
    assert 'duration_seconds' in features
    assert 'tags_count' in features
    assert 'description_length' in features
    
    print("✓ All features extraction works")


def test_sample_dataset_creation():
    """Test sample dataset creation."""
    df = create_sample_dataset(n_samples=100, output_path='/tmp/test_data.csv')
    
    assert len(df) == 100
    assert 'views' in df.columns
    assert df['views'].min() >= 0
    print("✓ Sample dataset creation works")


def test_model_training():
    """Test model training."""
    # Create small dataset
    df = create_sample_dataset(n_samples=100, output_path='/tmp/test_data.csv')
    
    feature_cols = [col for col in df.columns if col != 'views']
    X = df[feature_cols]
    y = df['views']
    
    # Train model
    predictor = YouTubeViewsPredictor(model_type='xgboost')
    results = predictor.train(X, y, test_size=0.2, random_state=42)
    
    # Check results
    assert 'train_metrics' in results
    assert 'test_metrics' in results
    assert 'mae' in results['test_metrics']
    assert 'r2' in results['test_metrics']
    
    print("✓ Model training works")
    print(f"  Test R²: {results['test_metrics']['r2']:.3f}")


def test_model_prediction():
    """Test model prediction."""
    # Create and train model
    df = create_sample_dataset(n_samples=100, output_path='/tmp/test_data.csv')
    feature_cols = [col for col in df.columns if col != 'views']
    X = df[feature_cols]
    y = df['views']
    
    predictor = YouTubeViewsPredictor(model_type='xgboost')
    predictor.train(X, y, test_size=0.2, random_state=42)
    
    # Make prediction
    test_features = X.iloc[0:1]
    prediction = predictor.predict(test_features)
    
    assert len(prediction) == 1
    assert prediction[0] >= 0
    
    print("✓ Model prediction works")
    print(f"  Sample prediction: {prediction[0]:,.0f} views")


def test_model_persistence():
    """Test model saving and loading."""
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create and train model
        df = create_sample_dataset(n_samples=100, output_path='/tmp/test_data.csv')
        feature_cols = [col for col in df.columns if col != 'views']
        X = df[feature_cols]
        y = df['views']
        
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        predictor.train(X, y, test_size=0.2, random_state=42)
        
        # Save model
        predictor.save_model(temp_dir)
        
        # Load model
        predictor2 = YouTubeViewsPredictor(model_type='xgboost')
        predictor2.load_model(temp_dir)
        
        # Test prediction with loaded model
        test_features = X.iloc[0:1]
        prediction = predictor2.predict(test_features)
        
        assert len(prediction) == 1
        print("✓ Model persistence works")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_optimal_features():
    """Test optimal features recommendations."""
    recommendations = get_optimal_features()
    
    assert 'title_recommendations' in recommendations
    assert 'temporal_recommendations' in recommendations
    assert 'duration_recommendations' in recommendations
    assert 'metadata_recommendations' in recommendations
    
    print("✓ Optimal features recommendations work")


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    # 1. Create feature extractor
    extractor = FeatureExtractor()
    
    # 2. Create sample video data
    video_data = {
        'title': 'Complete Python Tutorial for Beginners!',
        'duration': 720,  # 12 minutes
        'tags': 'python,tutorial,programming,coding,beginners',
        'publish_time': '2024-01-19 19:00:00',  # Friday 7 PM
        'description': 'Learn Python programming from scratch. Perfect for beginners! Subscribe for more tutorials.'
    }
    
    # 3. Extract features
    features = extractor.extract_all_features(video_data)
    features_df = pd.DataFrame([features])
    
    # 4. Load model (assuming it exists)
    try:
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        predictor.load_model('models')
        
        # 5. Make prediction
        prediction = predictor.predict(features_df)
        
        assert prediction[0] >= 0
        print("✓ End-to-end pipeline works")
        print(f"  Predicted views: {prediction[0]:,.0f}")
        
        # 6. Get top features
        top_features = predictor.get_top_features(n=5)
        print(f"  Top 5 features: {[f[0] for f in top_features]}")
        
    except FileNotFoundError:
        print("⚠ End-to-end test skipped (model not trained)")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running YouTube Views Predictor Tests")
    print("="*60 + "\n")
    
    tests = [
        test_feature_extractor_title,
        test_feature_extractor_temporal,
        test_feature_extractor_duration,
        test_feature_extractor_all,
        test_sample_dataset_creation,
        test_model_training,
        test_model_prediction,
        test_model_persistence,
        test_optimal_features,
        test_end_to_end_pipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
