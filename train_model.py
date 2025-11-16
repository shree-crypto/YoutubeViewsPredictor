"""
Train YouTube Views Prediction Model

This script trains the prediction model on sample data or real YouTube trending data
and saves it for use in the Streamlit app.

Usage:
    # Train with synthetic data (default)
    python train_model.py
    
    # Train with YouTube Trending dataset
    python train_model.py --dataset data/raw/youtube_trending.csv
    
    # Train with specific sample size
    python train_model.py --dataset data/raw/youtube_trending.csv --sample-size 5000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import argparse
from utils.model_training import YouTubeViewsPredictor, create_sample_dataset
from utils.feature_engineering import FeatureExtractor
from utils.data_loader import (
    load_youtube_trending_dataset,
    prepare_dataset_for_training,
    validate_dataset
)


def train_model(dataset_path=None, sample_size=None, synthetic_samples=1000):
    """
    Train and save the YouTube views prediction model.
    
    Args:
        dataset_path: Path to YouTube trending dataset CSV (None = use synthetic data)
        sample_size: Number of samples to use from dataset (None = use all)
        synthetic_samples: Number of synthetic samples to generate if using synthetic data
    """
    print("=" * 60)
    print("YouTube Views Predictor - Model Training")
    print("=" * 60)
    
    if dataset_path:
        # Load real YouTube trending dataset
        print(f"\n1. Loading YouTube Trending dataset from {dataset_path}...")
        try:
            raw_df = load_youtube_trending_dataset(
                dataset_path,
                sample_size=sample_size,
                remove_errors=True,
                min_views=0
            )
            
            # Validate dataset
            validation = validate_dataset(raw_df)
            if not validation['valid']:
                print("Dataset validation failed:")
                for error in validation['errors']:
                    print(f"  - {error}")
                return
            
            print(f"Dataset loaded successfully: {len(raw_df)} records")
            print(f"View count range: {raw_df['views'].min():,} - {raw_df['views'].max():,}")
            print(f"Mean views: {raw_df['views'].mean():,.0f}")
            
            # Extract features
            print("\n2. Extracting features from dataset...")
            feature_extractor = FeatureExtractor()
            X, y = prepare_dataset_for_training(raw_df, feature_extractor)
            
            print(f"Features extracted: {len(X)} samples with {len(X.columns)} features")
            print(f"Sample features: {X.columns.tolist()[:5]}... (showing first 5)")
            
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset_path}")
            print("Please ensure the file exists or use synthetic data (omit --dataset)")
            return
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
    else:
        # Create synthetic dataset
        print(f"\n1. Creating synthetic sample dataset ({synthetic_samples} samples)...")
        df = create_sample_dataset(n_samples=synthetic_samples)
        print(f"Dataset shape: {df.shape}")
        print(f"View count range: {df['views'].min():,} - {df['views'].max():,}")
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
    if dataset_path:
        print(f"Trained on: YouTube Trending dataset ({len(X)} samples)")
    else:
        print(f"Trained on: Synthetic data ({len(X)} samples)")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train YouTube Views Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with synthetic data (default)
  python train_model.py
  
  # Train with YouTube Trending dataset
  python train_model.py --dataset data/raw/US_youtube_trending_data.csv
  
  # Train with specific sample size
  python train_model.py --dataset data/raw/US_youtube_trending_data.csv --sample-size 5000
  
  # Train with more synthetic samples
  python train_model.py --synthetic-samples 5000
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to YouTube Trending dataset CSV file (default: use synthetic data)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of samples to use from the dataset (default: use all)'
    )
    
    parser.add_argument(
        '--synthetic-samples',
        type=int,
        default=1000,
        help='Number of synthetic samples to generate if not using real dataset (default: 1000)'
    )
    
    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        synthetic_samples=args.synthetic_samples
    )


if __name__ == '__main__':
    main()
