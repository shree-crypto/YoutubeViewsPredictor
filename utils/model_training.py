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


class YouTubeViewsPredictor:
    """Main predictor class for YouTube views."""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize predictor.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, X):
        """Prepare and scale features."""
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        return self.scaler.fit_transform(X)
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model.
        
        Args:
            X: Feature dataframe
            y: Target variable (view count)
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            dict: Training metrics
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
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
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
        
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
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature dataframe or dict
            
        Returns:
            Predicted view count
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Handle single prediction (dict input)
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Ensure correct feature order
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def save_model(self, model_dir='models'):
        """Save model and scaler to disk."""
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
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='models'):
        """Load model and scaler from disk."""
        model_path = os.path.join(model_dir, f'{self.model_type}_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        config_path = os.path.join(model_dir, 'model_config.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.feature_names = config['feature_names']
        self.feature_importance = config['feature_importance']
        
        print(f"Model loaded from {model_dir}")
    
    def get_top_features(self, n=10):
        """Get top N most important features."""
        if self.feature_importance is None:
            return None
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:n]


def create_sample_dataset(n_samples=1000, output_path='data/processed/sample_data.csv'):
    """Create a synthetic sample dataset for demonstration."""
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset created with {n_samples} samples")
    print(f"Saved to {output_path}")
    
    return df
