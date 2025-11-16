"""
Feature Engineering Module for YouTube Views Prediction

This module extracts and engineers features from video metadata including:
- Text features from title and description
- Temporal features from publish time
- Metadata features from video properties
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class FeatureExtractor:
    """Extract and engineer features from YouTube video data."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_title_features(self, title):
        """Extract features from video title."""
        if pd.isna(title) or not isinstance(title, str):
            return {
                'title_length': 0,
                'title_word_count': 0,
                'title_uppercase_ratio': 0,
                'title_digit_count': 0,
                'title_special_char_count': 0,
                'title_sentiment_polarity': 0,
                'title_sentiment_subjectivity': 0,
                'has_question_mark': 0,
                'has_exclamation': 0,
                'title_capitalized_words': 0
            }
        
        # Basic statistics
        title_length = len(title)
        words = title.split()
        word_count = len(words)
        
        # Character analysis
        uppercase_count = sum(1 for c in title if c.isupper())
        uppercase_ratio = uppercase_count / title_length if title_length > 0 else 0
        digit_count = sum(1 for c in title if c.isdigit())
        special_char_count = sum(1 for c in title if not c.isalnum() and not c.isspace())
        
        # Sentiment analysis
        blob = TextBlob(title)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
        
        # Special markers
        has_question_mark = 1 if '?' in title else 0
        has_exclamation = 1 if '!' in title else 0
        
        # Capitalized words (potential clickbait indicator)
        capitalized_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        return {
            'title_length': title_length,
            'title_word_count': word_count,
            'title_uppercase_ratio': uppercase_ratio,
            'title_digit_count': digit_count,
            'title_special_char_count': special_char_count,
            'title_sentiment_polarity': sentiment_polarity,
            'title_sentiment_subjectivity': sentiment_subjectivity,
            'has_question_mark': has_question_mark,
            'has_exclamation': has_exclamation,
            'title_capitalized_words': capitalized_words
        }
    
    def extract_temporal_features(self, publish_time):
        """Extract temporal features from publish time."""
        if pd.isna(publish_time):
            return {
                'publish_hour': 12,
                'publish_day_of_week': 3,
                'publish_month': 6,
                'is_weekend': 0,
                'is_peak_hour': 0
            }
        
        if isinstance(publish_time, str):
            try:
                dt = pd.to_datetime(publish_time)
            except:
                dt = datetime.now()
        else:
            dt = publish_time
        
        hour = dt.hour
        day_of_week = dt.dayofweek
        month = dt.month
        
        # Weekend indicator (Saturday=5, Sunday=6)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Peak hours (6-9 PM)
        is_peak_hour = 1 if 18 <= hour <= 21 else 0
        
        return {
            'publish_hour': hour,
            'publish_day_of_week': day_of_week,
            'publish_month': month,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak_hour
        }
    
    def extract_duration_features(self, duration_seconds):
        """Extract features from video duration."""
        if pd.isna(duration_seconds) or duration_seconds <= 0:
            duration_seconds = 300  # Default 5 minutes
        
        duration_minutes = duration_seconds / 60
        
        # Duration categories
        is_short = 1 if duration_seconds < 60 else 0  # < 1 min
        is_medium = 1 if 60 <= duration_seconds < 600 else 0  # 1-10 min
        is_long = 1 if duration_seconds >= 600 else 0  # >= 10 min
        
        return {
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_minutes,
            'is_short_video': is_short,
            'is_medium_video': is_medium,
            'is_long_video': is_long
        }
    
    def extract_tags_features(self, tags):
        """Extract features from video tags."""
        if pd.isna(tags) or not tags:
            return {
                'tags_count': 0,
                'avg_tag_length': 0
            }
        
        if isinstance(tags, str):
            # Assume tags are separated by comma or pipe
            tag_list = [tag.strip() for tag in re.split('[,|]', tags)]
        elif isinstance(tags, list):
            tag_list = tags
        else:
            tag_list = []
        
        tags_count = len(tag_list)
        avg_tag_length = np.mean([len(tag) for tag in tag_list]) if tag_list else 0
        
        return {
            'tags_count': tags_count,
            'avg_tag_length': avg_tag_length
        }
    
    def extract_description_features(self, description):
        """Extract features from video description."""
        if pd.isna(description) or not isinstance(description, str):
            return {
                'description_length': 0,
                'description_word_count': 0,
                'description_link_count': 0
            }
        
        description_length = len(description)
        word_count = len(description.split())
        
        # Count URLs in description
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        link_count = len(re.findall(url_pattern, description))
        
        return {
            'description_length': description_length,
            'description_word_count': word_count,
            'description_link_count': link_count
        }
    
    def extract_all_features(self, data_dict):
        """Extract all features from a video data dictionary."""
        features = {}
        
        # Title features
        if 'title' in data_dict:
            features.update(self.extract_title_features(data_dict['title']))
        
        # Temporal features
        if 'publish_time' in data_dict:
            features.update(self.extract_temporal_features(data_dict['publish_time']))
        
        # Duration features
        if 'duration' in data_dict:
            features.update(self.extract_duration_features(data_dict['duration']))
        
        # Tags features
        if 'tags' in data_dict:
            features.update(self.extract_tags_features(data_dict['tags']))
        
        # Description features
        if 'description' in data_dict:
            features.update(self.extract_description_features(data_dict['description']))
        
        return features
    
    def process_dataframe(self, df):
        """Process entire dataframe and extract features."""
        feature_dicts = []
        
        for idx, row in df.iterrows():
            data_dict = row.to_dict()
            features = self.extract_all_features(data_dict)
            feature_dicts.append(features)
        
        features_df = pd.DataFrame(feature_dicts)
        return features_df


def get_optimal_features():
    """Return recommended optimal features for high view count."""
    return {
        'title_recommendations': [
            'Keep title length between 50-70 characters',
            'Use 1-2 capitalized words for emphasis',
            'Include numbers or statistics',
            'Add question marks or exclamation points for engagement',
            'Use emotionally charged words (but maintain authenticity)'
        ],
        'temporal_recommendations': [
            'Upload during peak hours (6-9 PM local time)',
            'Friday and weekend uploads tend to perform better',
            'Avoid Monday mornings',
            'Consider seasonal trends (summer vs. holidays)'
        ],
        'duration_recommendations': [
            'Optimal duration: 7-15 minutes for most content',
            'Shorter videos (< 5 min) for quick tips/news',
            'Longer videos (> 15 min) for in-depth tutorials/entertainment',
            'Match duration to content type and audience retention'
        ],
        'metadata_recommendations': [
            'Use 10-15 relevant tags',
            'Write detailed descriptions (200-300 words)',
            'Include 2-3 links in description',
            'Choose the most relevant category',
            'Add timestamps for longer videos'
        ]
    }
