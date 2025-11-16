# YouTube Views Predictor - Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Training the Model](#training-the-model)
3. [Using the Web Interface](#using-the-web-interface)
4. [Programmatic Usage](#programmatic-usage)
5. [Understanding Predictions](#understanding-predictions)
6. [Optimization Strategies](#optimization-strategies)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB free disk space
- Internet connection (for initial setup)

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/shree-crypto/YoutubeViewsPredictor.git
cd YoutubeViewsPredictor
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data processing)
- scikit-learn (ML utilities)
- xgboost, lightgbm (models)
- streamlit (web UI)
- nltk, textblob (NLP)
- plotly (visualizations)

---

## Training the Model

### Basic Training

Run the training script to create and train a model:

```bash
python train_model.py
```

**What happens:**
1. Generates synthetic dataset (1000 video samples)
2. Extracts 25+ features from video data
3. Trains XGBoost model with optimized hyperparameters
4. Evaluates model performance (train/test split)
5. Displays feature importance
6. Saves model to `models/` directory

**Expected output:**
```
Training Metrics:
  MAE: 470.95
  RMSE: 680.77
  R2: 1.00

Test Metrics:
  MAE: 16901.24
  RMSE: 21005.05
  R2: 0.69
```

### Using Your Own Data

To train on custom YouTube data:

1. **Prepare your CSV file** with these columns:
```
title,duration,tags,publish_time,description,views
```

2. **Load and train**:
```python
import pandas as pd
from utils.feature_engineering import FeatureExtractor
from utils.model_training import YouTubeViewsPredictor

# Load your data
df = pd.read_csv('your_data.csv')

# Extract features
extractor = FeatureExtractor()
features_df = extractor.process_dataframe(df)

# Train model
predictor = YouTubeViewsPredictor(model_type='xgboost')
results = predictor.train(features_df, df['views'])

# Save model
predictor.save_model('models')
```

---

## Using the Web Interface

### Launching the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### Interface Sections

#### 1. Prediction Tab 🎯

**Input Fields:**

- **Video Title**: Your video's title
  - Example: "How to Build Amazing Projects in 2024"
  - Tips: 50-70 characters optimal
  
- **Description**: Video description
  - Example: "Learn step-by-step how to..."
  - Tips: 200-300 words recommended
  
- **Tags**: Comma-separated keywords
  - Example: "tutorial,programming,python,beginners"
  - Tips: Use 10-15 relevant tags
  
- **Duration**: Video length in minutes
  - Example: 10.0
  - Tips: 7-15 minutes optimal for most content
  
- **Publish Date/Time**: When you'll upload
  - Tips: Friday-Sunday, 6-9 PM best
  
- **Category**: Video category
  - Choose most relevant

**Options:**
- ☑️ Show confidence interval: Displays prediction range
- ☑️ Compare with optimal: Shows improvement suggestions

**Output:**
- **Predicted Views**: Main prediction
- **Confidence Interval**: Range (80%-120%)
- **Optimization Suggestions**: Specific improvements
- **Potential Increase**: Expected improvement percentage

#### 2. Feature Importance Tab 📊

Shows which features most influence predictions:
- Bar chart of top 15 features
- Categorized by type (title, temporal, video)
- Helps understand what drives views

**Key Insights:**
- Peak hour publishing is most important
- Weekend uploads get more views
- Question marks in titles help
- Duration optimization matters

#### 3. Recommendations Tab 💡

Evidence-based optimization advice:
- **Title optimization**: Structure and keywords
- **Publishing time**: Best days and hours
- **Duration**: Optimal length for content type
- **Metadata**: Tags, descriptions, links

---

## Programmatic Usage

### Basic Prediction

```python
from utils.model_training import YouTubeViewsPredictor
from utils.feature_engineering import FeatureExtractor

# Load model
predictor = YouTubeViewsPredictor(model_type='xgboost')
predictor.load_model('models')

# Create feature extractor
extractor = FeatureExtractor()

# Your video data
video_data = {
    'title': 'Amazing Python Tutorial!',
    'duration': 600,  # 10 minutes in seconds
    'tags': 'python,tutorial,programming',
    'publish_time': '2024-01-15 18:00:00',
    'description': 'Learn Python in this comprehensive tutorial...'
}

# Extract features
features = extractor.extract_all_features(video_data)

# Predict
views = predictor.predict(features)
print(f"Predicted views: {views[0]:,.0f}")
```

### Batch Predictions

```python
import pandas as pd

# Load multiple videos
videos_df = pd.read_csv('videos.csv')

# Extract features for all
features_df = extractor.process_dataframe(videos_df)

# Predict all at once
predictions = predictor.predict(features_df)

# Add to dataframe
videos_df['predicted_views'] = predictions
```

### Feature Analysis

```python
# Get feature importance
top_features = predictor.get_top_features(n=10)

for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
```

### Get Optimization Recommendations

```python
from utils.feature_engineering import get_optimal_features

recommendations = get_optimal_features()

# Print title recommendations
for rec in recommendations['title_recommendations']:
    print(f"- {rec}")
```

---

## Understanding Predictions

### What the Numbers Mean

**Predicted Views**: Expected view count based on input parameters
- Actual results may vary ±20-30%
- Depends on content quality, thumbnail, promotion
- Based on historical patterns

**Confidence Interval**:
- Lower bound (80%): Conservative estimate
- Upper bound (120%): Optimistic estimate
- Wider interval = higher uncertainty

### Factors Affecting Accuracy

**Positive factors** (improve accuracy):
- Using real YouTube data for training
- Larger training dataset
- Recent, relevant data
- Complete feature information

**Limitations**:
- Cannot account for: content quality, thumbnail, creator reputation
- External factors: trending topics, algorithm changes
- Viral potential is unpredictable
- Channel-specific factors not included

---

## Optimization Strategies

### Quick Wins (Easy to Implement)

1. **Publishing Time**:
   - Upload 6-9 PM (18:00-21:00) local time
   - Friday-Sunday preferred
   - Avoid Monday mornings

2. **Title Optimization**:
   - Keep 50-70 characters
   - Add "?" or "!" for engagement
   - Include numbers (e.g., "Top 5...", "In 10 Minutes")
   - Use emotional words

3. **Tags**:
   - Use 10-15 relevant tags
   - Mix broad and specific terms
   - Include variations and synonyms

### Advanced Strategies

1. **Duration Optimization**:
   - Quick tips: 3-5 minutes
   - Tutorials: 7-15 minutes
   - Deep dives: 15-30 minutes
   - Match audience retention

2. **Description Optimization**:
   - Write 200-300 words
   - Front-load important keywords
   - Include 2-3 relevant links
   - Add timestamps for longer videos

3. **Seasonal Timing**:
   - Consider holidays, trends
   - Summer vs. school year
   - Industry-specific cycles

### Testing Your Changes

Use the app to compare scenarios:
1. Input current parameters → Get baseline prediction
2. Modify one parameter → Compare new prediction
3. Apply multiple optimizations → See cumulative effect

---

## Troubleshooting

### Model Not Found Error

**Error**: "Model not found! Please run train_model.py first."

**Solution**:
```bash
python train_model.py
```

This creates the model files in `models/` directory.

### Import Errors

**Error**: "ModuleNotFoundError: No module named 'xgboost'"

**Solution**:
```bash
pip install -r requirements.txt
```

### NLTK Data Missing

**Error**: "Resource stopwords not found"

**Solution**:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Streamlit Port Already in Use

**Error**: "Address already in use"

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill
```

### Low Prediction Accuracy

**Issue**: Predictions seem inaccurate

**Solutions**:
1. Train on real YouTube data (not synthetic)
2. Use larger dataset (1000+ videos)
3. Include recent videos (last 6-12 months)
4. Verify input data quality

### Performance Issues

**Issue**: Slow predictions or training

**Solutions**:
1. Use LightGBM instead of XGBoost:
   ```python
   predictor = YouTubeViewsPredictor(model_type='lightgbm')
   ```
2. Reduce n_estimators in model config
3. Use smaller dataset for training

---

## Best Practices

### Data Collection
- Collect diverse video types
- Include various channels and sizes
- Keep data recent (< 1 year old)
- Verify data quality

### Model Training
- Retrain periodically (monthly)
- Use 80/20 train/test split
- Monitor performance metrics
- Keep track of feature importance changes

### Making Predictions
- Provide complete information
- Use realistic parameters
- Consider multiple scenarios
- Don't rely solely on predictions

### Optimization
- Test one change at a time
- Track actual results
- Iterate based on feedback
- Adapt to your specific audience

---

## Next Steps

1. **Collect Real Data**: Use YouTube API to gather actual video data
2. **Retrain Model**: Use your data for better predictions
3. **A/B Testing**: Test optimization strategies
4. **Track Results**: Compare predictions with actual views
5. **Iterate**: Refine based on learnings

---

## Support

- **Documentation**: See README.md
- **Issues**: https://github.com/shree-crypto/YoutubeViewsPredictor/issues
- **Examples**: Check notebooks/ directory

---

**Happy optimizing! 📈**
