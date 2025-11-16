"""
Tests for YouTube Dataset Loading Module

Run with: python -m pytest test_data_loader.py -v
Or simply: python test_data_loader.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pytest
import pandas as pd
from utils.data_loader import (
    parse_iso_duration,
    load_youtube_trending_dataset,
    prepare_dataset_for_training,
    validate_dataset
)
from utils.feature_engineering import FeatureExtractor


def test_parse_iso_duration_basic():
    """Test basic ISO 8601 duration parsing."""
    assert parse_iso_duration('PT15M30S') == 930
    assert parse_iso_duration('PT1H2M10S') == 3730
    assert parse_iso_duration('PT45S') == 45
    assert parse_iso_duration('PT1H') == 3600
    assert parse_iso_duration('PT30M') == 1800
    print("✓ ISO duration parsing works")


def test_parse_iso_duration_edge_cases():
    """Test edge cases for ISO duration parsing."""
    assert parse_iso_duration('PT0S') == 0
    assert parse_iso_duration('') == 0
    assert parse_iso_duration(None) == 0
    print("✓ ISO duration edge cases handled")


def test_load_youtube_trending_dataset():
    """Test loading YouTube trending dataset."""
    dataset_path = 'data/raw/sample_youtube_trending.csv'
    
    if not os.path.exists(dataset_path):
        print("⚠ Sample dataset not found, skipping test")
        return
    
    df = load_youtube_trending_dataset(dataset_path)
    
    # Check required columns
    required_cols = ['title', 'duration', 'tags', 'publish_time', 'description', 'views']
    assert all(col in df.columns for col in required_cols)
    
    # Check data types
    assert df['views'].dtype in [int, 'int64']
    assert df['duration'].dtype in [int, 'int64']
    
    # Check no negative values
    assert (df['views'] >= 0).all()
    assert (df['duration'] >= 0).all()
    
    print(f"✓ YouTube dataset loaded: {len(df)} records")


def test_load_with_sample_size():
    """Test loading dataset with sample size limit."""
    dataset_path = 'data/raw/sample_youtube_trending.csv'
    
    if not os.path.exists(dataset_path):
        print("⚠ Sample dataset not found, skipping test")
        return
    
    df = load_youtube_trending_dataset(dataset_path, sample_size=5)
    assert len(df) == 5
    print("✓ Sample size limiting works")


def test_validate_dataset():
    """Test dataset validation."""
    # Create a valid test dataset
    df = pd.DataFrame({
        'title': ['Test Video 1', 'Test Video 2'],
        'duration': [600, 900],
        'tags': ['tag1,tag2', 'tag3,tag4'],
        'publish_time': ['2024-01-15 18:00:00', '2024-01-15 19:00:00'],
        'description': ['Description 1', 'Description 2'],
        'views': [1000, 2000]
    })
    
    validation = validate_dataset(df)
    
    assert validation['valid'] == True
    assert validation['num_records'] == 2
    assert 'statistics' in validation
    assert validation['statistics']['view_count']['min'] == 1000
    assert validation['statistics']['view_count']['max'] == 2000
    
    print("✓ Dataset validation works")


def test_validate_dataset_missing_columns():
    """Test validation with missing columns."""
    # Create dataset with missing columns
    df = pd.DataFrame({
        'title': ['Test Video'],
        'views': [1000]
    })
    
    validation = validate_dataset(df)
    
    assert validation['valid'] == False
    assert len(validation['errors']) > 0
    
    print("✓ Missing column detection works")


def test_prepare_dataset_for_training():
    """Test preparing dataset for training."""
    dataset_path = 'data/raw/sample_youtube_trending.csv'
    
    if not os.path.exists(dataset_path):
        print("⚠ Sample dataset not found, skipping test")
        return
    
    df = load_youtube_trending_dataset(dataset_path)
    extractor = FeatureExtractor()
    
    X, y = prepare_dataset_for_training(df, extractor)
    
    # Check output shapes
    assert len(X) == len(y)
    assert len(X) > 0
    
    # Check features are extracted
    assert 'title_length' in X.columns
    assert 'publish_hour' in X.columns
    assert 'duration_seconds' in X.columns
    
    # Check target values
    assert (y > 0).all()
    
    print(f"✓ Dataset preparation works: {len(X)} samples, {len(X.columns)} features")


def test_end_to_end_youtube_dataset():
    """Test complete pipeline with YouTube dataset."""
    dataset_path = 'data/raw/sample_youtube_trending.csv'
    
    if not os.path.exists(dataset_path):
        print("⚠ Sample dataset not found, skipping test")
        return
    
    from utils.model_training import YouTubeViewsPredictor
    
    # Load dataset
    df = load_youtube_trending_dataset(dataset_path, sample_size=10)
    
    # Extract features
    extractor = FeatureExtractor()
    X, y = prepare_dataset_for_training(df, extractor)
    
    # Train model (only if we have enough data)
    if len(X) >= 5:
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        results = predictor.train(X, y, test_size=0.2, random_state=42)
        
        assert 'train_metrics' in results
        assert 'test_metrics' in results
        assert results['train_metrics']['r2'] > 0  # Some positive correlation
        
        print("✓ End-to-end pipeline with YouTube dataset works")
    else:
        print("⚠ Not enough data for training test")


# Run tests if executed directly
if __name__ == '__main__':
    print("Running YouTube Data Loader Tests...")
    print("=" * 60)
    
    test_parse_iso_duration_basic()
    test_parse_iso_duration_edge_cases()
    test_load_youtube_trending_dataset()
    test_load_with_sample_size()
    test_validate_dataset()
    test_validate_dataset_missing_columns()
    test_prepare_dataset_for_training()
    test_end_to_end_youtube_dataset()
    
    print("=" * 60)
    print("All tests passed! ✓")
