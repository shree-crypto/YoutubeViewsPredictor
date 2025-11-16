"""
Data Loading Module for YouTube Dataset

Provides functionality to load and preprocess the Trending YouTube Video Statistics dataset.
Supports various YouTube dataset formats including Kaggle datasets.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_iso_duration(duration_str: str) -> int:
    """
    Parse ISO 8601 duration format (e.g., PT15M33S, PT1H2M10S) to seconds.
    
    Args:
        duration_str: ISO 8601 duration string (e.g., 'PT15M33S')
        
    Returns:
        Duration in seconds
        
    Examples:
        >>> parse_iso_duration('PT15M33S')
        933
        >>> parse_iso_duration('PT1H2M10S')
        3730
        >>> parse_iso_duration('PT45S')
        45
    """
    if pd.isna(duration_str) or not duration_str:
        return 0
    
    # Remove 'PT' prefix
    duration_str = str(duration_str).replace('PT', '')
    
    hours = 0
    minutes = 0
    seconds = 0
    
    # Extract hours
    hour_match = re.search(r'(\d+)H', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))
    
    # Extract minutes
    minute_match = re.search(r'(\d+)M', duration_str)
    if minute_match:
        minutes = int(minute_match.group(1))
    
    # Extract seconds
    second_match = re.search(r'(\d+)S', duration_str)
    if second_match:
        seconds = int(second_match.group(1))
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def load_youtube_trending_dataset(
    file_path: str,
    sample_size: Optional[int] = None,
    remove_errors: bool = True,
    min_views: int = 0
) -> pd.DataFrame:
    """
    Load and preprocess the Trending YouTube Video Statistics dataset.
    
    This function handles the Kaggle Trending YouTube dataset format and
    maps it to the format expected by the feature engineering pipeline.
    
    Args:
        file_path: Path to the CSV file containing YouTube trending data
        sample_size: Optional number of samples to randomly select (None = all)
        remove_errors: Whether to remove videos marked as errors
        min_views: Minimum view count threshold (default: 0)
        
    Returns:
        DataFrame with columns: title, duration, tags, publish_time, description, views
        
    Raises:
        FileNotFoundError: If file_path doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    logger.info(f"Loading YouTube trending dataset from {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        logger.info(f"Loaded {len(df)} records from dataset")
        logger.info(f"Columns found: {df.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise
    
    # Check for required columns
    required_cols = ['title', 'views']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove error videos if specified
    if remove_errors and 'video_error_or_removed' in df.columns:
        original_len = len(df)
        df = df[df['video_error_or_removed'] == False].copy()
        logger.info(f"Removed {original_len - len(df)} error videos")
    
    # Filter by minimum views
    if min_views > 0:
        original_len = len(df)
        df = df[df['views'] >= min_views].copy()
        logger.info(f"Filtered to {len(df)} videos with >= {min_views} views")
    
    # Handle duration column - could be in different formats
    if 'duration' in df.columns:
        # Check if duration is in ISO format (contains 'PT')
        if df['duration'].astype(str).str.contains('PT').any():
            logger.info("Converting ISO 8601 duration format to seconds")
            df['duration'] = df['duration'].apply(parse_iso_duration)
        # If duration is already numeric, keep it
        elif pd.api.types.is_numeric_dtype(df['duration']):
            df['duration'] = df['duration'].fillna(0).astype(int)
        else:
            logger.warning("Duration format not recognized, setting to default 600 seconds")
            df['duration'] = 600
    else:
        # Default duration if not available
        logger.info("Duration column not found, using default 600 seconds (10 minutes)")
        df['duration'] = 600
    
    # Handle tags column
    if 'tags' in df.columns:
        # Replace NaN with empty string and convert to string
        df['tags'] = df['tags'].fillna('').astype(str)
        # Clean up tags format if needed
        df['tags'] = df['tags'].str.replace('[none]', '', case=False)
        df['tags'] = df['tags'].str.replace('|', ',')  # Some datasets use | separator
    else:
        logger.info("Tags column not found, using empty tags")
        df['tags'] = ''
    
    # Handle publish_time column
    publish_time_cols = ['publish_time', 'publishedAt', 'published_at', 'publishTime']
    publish_col = None
    for col in publish_time_cols:
        if col in df.columns:
            publish_col = col
            break
    
    if publish_col:
        df['publish_time'] = df[publish_col]
        # Ensure it's in string format
        df['publish_time'] = df['publish_time'].astype(str)
    else:
        logger.warning("Publish time column not found, using current timestamp")
        df['publish_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Handle description column
    if 'description' not in df.columns:
        logger.info("Description column not found, using empty descriptions")
        df['description'] = ''
    else:
        df['description'] = df['description'].fillna('').astype(str)
    
    # Select and rename columns to match expected format
    output_df = pd.DataFrame({
        'title': df['title'].astype(str),
        'duration': df['duration'].astype(int),
        'tags': df['tags'],
        'publish_time': df['publish_time'],
        'description': df['description'],
        'views': df['views'].astype(int)
    })
    
    # Remove any rows with missing critical data
    output_df = output_df.dropna(subset=['title', 'views'])
    
    # Sample if requested
    if sample_size and sample_size < len(output_df):
        logger.info(f"Sampling {sample_size} records from dataset")
        output_df = output_df.sample(n=sample_size, random_state=42)
    
    logger.info(f"Preprocessed dataset: {len(output_df)} records")
    logger.info(f"View count range: {output_df['views'].min()} - {output_df['views'].max()}")
    logger.info(f"Mean views: {output_df['views'].mean():.0f}")
    
    return output_df.reset_index(drop=True)


def prepare_dataset_for_training(df: pd.DataFrame, feature_extractor) -> tuple:
    """
    Prepare a loaded dataset for model training by extracting features.
    
    Args:
        df: DataFrame with columns: title, duration, tags, publish_time, description, views
        feature_extractor: FeatureExtractor instance
        
    Returns:
        tuple: (X, y) where X is feature DataFrame and y is target Series
    """
    logger.info("Extracting features from dataset")
    
    features_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            data = {
                'title': row['title'],
                'duration': row['duration'],
                'tags': row['tags'],
                'publish_time': row['publish_time'],
                'description': row['description']
            }
            
            features = feature_extractor.extract_all_features(data)
            features_list.append(features)
            valid_indices.append(idx)
            
        except Exception as e:
            logger.warning(f"Error extracting features for row {idx}: {e}")
            continue
    
    # Create feature DataFrame
    X = pd.DataFrame(features_list)
    y = df.loc[valid_indices, 'views'].reset_index(drop=True)
    
    logger.info(f"Extracted {len(X)} feature sets with {len(X.columns)} features each")
    
    return X, y


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a YouTube dataset and return statistics.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results and statistics
    """
    validation_results = {
        'valid': True,
        'num_records': len(df),
        'errors': []
    }
    
    # Check required columns
    required_cols = ['title', 'duration', 'tags', 'publish_time', 'description', 'views']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if validation_results['valid']:
        # Calculate statistics
        validation_results['statistics'] = {
            'total_records': len(df),
            'view_count': {
                'min': int(df['views'].min()),
                'max': int(df['views'].max()),
                'mean': float(df['views'].mean()),
                'median': float(df['views'].median())
            },
            'duration': {
                'min': int(df['duration'].min()),
                'max': int(df['duration'].max()),
                'mean': float(df['duration'].mean())
            },
            'missing_data': {
                'title': int(df['title'].isna().sum()),
                'description': int(df['description'].isna().sum()),
                'tags': int(df['tags'].isna().sum())
            }
        }
    
    return validation_results
