"""
Train YouTube Views Prediction Model

This script trains the prediction model on sample data and saves it for use in the Streamlit app.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.model_training import YouTubeViewsPredictor, create_sample_dataset
from utils.feature_engineering import FeatureExtractor


def train_model():
    """Train and save the YouTube views prediction model."""
    print("=" * 60)
    print("YouTube Views Predictor - Model Training")
    print("=" * 60)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    df = create_sample_dataset(n_samples=1000)
    print(f"Dataset shape: {df.shape}")
    print(f"View count range: {df['views'].min()} - {df['views'].max()}")
    print(f"Mean views: {df['views'].mean():.0f}")
    
    # Prepare features and target
    print("\n2. Preparing features and target...")
    feature_cols = [col for col in df.columns if col != 'views']
    X = df[feature_cols]
    y = df['views']
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    # Train model
    print("\n3. Training XGBoost model...")
    predictor = YouTubeViewsPredictor(model_type='xgboost')
    results = predictor.train(X, y, test_size=0.2, random_state=42)
    
    # Display results
    print("\n4. Training Results:")
    print("-" * 60)
    print("Training Metrics:")
    for metric, value in results['train_metrics'].items():
        print(f"  {metric.upper()}: {value:.2f}")
    
    print("\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric.upper()}: {value:.2f}")
    
    # Top features
    print("\n5. Top 10 Most Important Features:")
    print("-" * 60)
    top_features = predictor.get_top_features(n=10)
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Save model
    print("\n6. Saving model...")
    predictor.save_model(model_dir='models')
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("Model saved to 'models/' directory")
    print("=" * 60)


if __name__ == '__main__':
    train_model()
