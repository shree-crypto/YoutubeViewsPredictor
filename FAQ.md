# Frequently Asked Questions (FAQ)

Common questions about the YouTube Views Predictor.

## Table of Contents
- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Usage Questions](#usage-questions)
- [Model Performance](#model-performance)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## General Questions

### What is the YouTube Views Predictor?

The YouTube Views Predictor is a machine learning tool that predicts how many views a YouTube video is likely to receive based on its metadata (title, description, tags, duration, publishing time, etc.). It uses XGBoost, a gradient boosting algorithm, trained on video features.

### How accurate are the predictions?

Accuracy depends on:
- **Training data quality**: Real YouTube data yields better results than synthetic data
- **Dataset size**: Larger datasets (1000+ videos) improve accuracy
- **Data recency**: Recent data reflects current YouTube algorithm behavior

With synthetic data: R² score ~0.69 (explains 69% of variance)
With real data: Can achieve R² > 0.80

**Note**: Predictions are estimates. Actual views depend on many factors including content quality, thumbnail, promotion, and viral potential.

### Is this free to use?

Yes! The project is open-source under the MIT License. You can use, modify, and distribute it freely.

### Do I need a YouTube API key?

No, not for basic usage. The tool works with synthetic data out of the box. However, for better predictions on your specific content, you can:
- Train on real YouTube data (requires YouTube API key)
- Use publicly available datasets (Kaggle, etc.)

### Can this guarantee my video will get X views?

No. This tool provides **predictions based on historical patterns**, not guarantees. Actual performance depends on:
- Content quality
- Thumbnail appeal
- Audience engagement
- Promotion efforts
- Algorithm changes
- Trending topics
- Competition

### What makes a good YouTube video according to this model?

Based on feature importance analysis:
1. **Publishing during peak hours** (6-9 PM) - Most important
2. **Weekend uploads** - Significantly increases views
3. **Title with question marks** - Drives engagement
4. **Optimal duration** (7-15 minutes)
5. **10-15 relevant tags**
6. **Comprehensive descriptions**

---

## Installation & Setup

### What are the system requirements?

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows, macOS, or Linux

### Why do I get "torch" installation errors?

PyTorch (`torch`) is listed in requirements.txt but is optional for basic functionality. If installation fails:

```bash
# Install without torch
pip install pandas numpy scikit-learn xgboost lightgbm streamlit plotly nltk textblob joblib
```

You can use the predictor without PyTorch. It's only needed for advanced deep learning features (not yet implemented).

### Can I use this in a virtual environment?

Yes! We strongly recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### How long does training take?

- **Synthetic dataset (1000 samples)**: ~10-30 seconds
- **Real dataset (10,000 samples)**: ~1-2 minutes
- **Large dataset (100,000+ samples)**: ~10-30 minutes

Training time depends on CPU speed and dataset size.

---

## Usage Questions

### Do I need to train the model every time?

No! Train once and the model is saved to the `models/` directory. It will be loaded automatically by the app.

**Retrain when:**
- You have new/better training data
- YouTube algorithm changes significantly
- Model performance degrades
- Monthly retraining is recommended

### How do I use my own YouTube data?

**Option 1: YouTube API**
```python
# Use YouTube Data API v3
# Fetch video statistics and save to CSV
```

**Option 2: Kaggle Datasets**
- Download YouTube datasets from Kaggle
- Place in `data/raw/` directory
- Use the training script with your data

**Required CSV format:**
```
title,duration,tags,publish_time,description,views
"Video Title",600,"tag1,tag2","2024-01-15 18:00:00","Description",10000
```

### Can I predict views for videos in other languages?

The current implementation works best with English content because:
- Sentiment analysis uses English language models
- Training data is typically English

For other languages:
- The model will still work (structural features are language-agnostic)
- Text analysis features may be less accurate
- Consider training on language-specific data

### Can I batch process multiple videos?

Yes! See [EXAMPLES.md](EXAMPLES.md) for batch processing examples:

```python
import pandas as pd
videos_df = pd.read_csv('my_videos.csv')
features_df = extractor.process_dataframe(videos_df)
predictions = predictor.predict(features_df)
```

### How do I export predictions?

```python
import pandas as pd

# After predictions
results_df = pd.DataFrame({
    'title': titles,
    'predicted_views': predictions
})
results_df.to_csv('predictions.csv', index=False)
```

Or use the Streamlit app's export feature (if implemented).

---

## Model Performance

### Why are predictions sometimes inaccurate?

Common reasons:
1. **Training data mismatch**: Model trained on different content type/niche
2. **Synthetic data**: Using sample data instead of real YouTube data
3. **Missing features**: Thumbnail quality, creator reputation not included
4. **Algorithm changes**: YouTube's algorithm evolves over time
5. **Viral content**: Unpredictable by nature
6. **External factors**: Trending topics, seasonal effects

### What's a good R² score?

- **R² > 0.7**: Good predictive power
- **R² 0.5-0.7**: Moderate predictive power
- **R² < 0.5**: Poor predictive power, consider retraining

The synthetic dataset achieves R² ~0.69, which is reasonable for demonstration.

### How can I improve prediction accuracy?

1. **Use real YouTube data** instead of synthetic
2. **Increase training dataset size** (1000+ videos minimum)
3. **Use recent data** (last 6-12 months)
4. **Include your content niche** in training data
5. **Retrain regularly** (monthly)
6. **Add custom features** relevant to your niche
7. **Tune hyperparameters** for your specific data

### Why do some features matter more than others?

Feature importance is learned from data patterns. In our model:

**High importance:**
- `is_peak_hour` (62%): Publishing time strongly affects initial visibility
- `is_weekend` (13%): Weekend algorithm behavior differs significantly

**Lower importance:**
- Individual text features: Contribute less individually but matter collectively

### Can I add custom features?

Yes! Edit `utils/feature_engineering.py`:

```python
def extract_custom_features(self, data):
    features = {}
    # Your custom feature extraction
    features['has_tutorial_in_title'] = 'tutorial' in data['title'].lower()
    # Add more...
    return features
```

Then retrain the model with your enhanced features.

---

## Troubleshooting

### "Model not found" error

**Cause**: Model hasn't been trained yet.

**Solution**:
```bash
python train_model.py
```

### "ModuleNotFoundError" errors

**Cause**: Missing dependencies.

**Solution**:
```bash
pip install -r requirements.txt
```

### "NLTK data missing" error

**Cause**: NLTK corpora not downloaded.

**Solution**:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Streamlit app won't start

**Common issues:**

1. **Port already in use**:
```bash
streamlit run app.py --server.port 8502
```

2. **Model not loaded**:
```bash
python train_model.py
```

3. **Import errors**:
```bash
pip install streamlit
```

### Predictions are all the same

**Possible causes:**
1. Model not properly trained
2. Features not varying
3. Scaler issues

**Solution**:
```bash
# Retrain from scratch
rm -rf models/*
python train_model.py
```

### App runs slowly

**Solutions:**
1. **Use LightGBM** (faster than XGBoost):
```python
predictor = YouTubeViewsPredictor(model_type='lightgbm')
```

2. **Reduce model complexity** in `utils/model_training.py`:
```python
n_estimators=100,  # Reduce from 200
max_depth=6,       # Reduce from 8
```

3. **Use smaller training dataset**

### Memory errors during training

**Solutions:**
1. Reduce dataset size:
```python
df = create_sample_dataset(n_samples=500)  # Instead of 1000
```

2. Use LightGBM (more memory efficient)
3. Close other applications
4. Increase system swap space

---

## Advanced Usage

### Can I use this in production?

Yes, but consider:
1. **Retrain regularly** with fresh data
2. **Monitor prediction accuracy**
3. **Set up logging** and error handling
4. **Use a proper database** for storing predictions
5. **Implement caching** for performance
6. **Add rate limiting** if exposing as API
7. **Consider hosting** (AWS, GCP, Azure)

### How do I deploy as a web service?

**Option 1: Streamlit Sharing** (easiest)
```bash
# Push to GitHub
# Connect to Streamlit Sharing
# Deploy automatically
```

**Option 2: Docker**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

**Option 3: Cloud Hosting**
- Deploy to Heroku, AWS, or Google Cloud
- Use gunicorn + Streamlit for production

### Can I create an API instead of using Streamlit?

Yes! Create a FastAPI or Flask API:

```python
from fastapi import FastAPI
from utils.model_training import YouTubeViewsPredictor
from utils.feature_engineering import FeatureExtractor

app = FastAPI()
predictor = YouTubeViewsPredictor(model_type='xgboost')
predictor.load_model('models')
extractor = FeatureExtractor()

@app.post("/predict")
def predict(video_data: dict):
    features = extractor.extract_all_features(video_data)
    prediction = predictor.predict(features)[0]
    return {"predicted_views": prediction}
```

### How do I integrate with my content management system?

Example integration:

```python
# In your CMS
from youtube_views_predictor import YouTubeViewsPredictor, FeatureExtractor

class VideoManager:
    def __init__(self):
        self.predictor = YouTubeViewsPredictor()
        self.predictor.load_model('path/to/models')
        self.extractor = FeatureExtractor()
    
    def predict_before_publish(self, video_data):
        features = self.extractor.extract_all_features(video_data)
        return self.predictor.predict(features)[0]
```

### Can I use this for other platforms (TikTok, Instagram)?

The core approach works for any video platform, but you'll need to:
1. Collect platform-specific data
2. Adjust feature engineering for platform characteristics
3. Retrain model on new platform data
4. Account for platform-specific algorithms

### How do I implement A/B testing?

See [EXAMPLES.md](EXAMPLES.md) for A/B testing examples. Basic approach:

1. Create variations of your video parameters
2. Predict views for each variation
3. Choose the best-performing variant
4. After publishing, compare actual vs predicted
5. Iterate and improve

### Can I use deep learning models?

Yes! The architecture supports alternative models. To implement:

1. Add BERT for title/description embeddings
2. Add CNN for thumbnail analysis (requires image data)
3. Add LSTM for temporal patterns
4. Implement in `utils/model_training.py`

Example:
```python
from transformers import BertModel
# Implement custom model class
# Update training pipeline
```

---

## Still Have Questions?

- **Check the documentation**: [README.md](README.md), [GETTING_STARTED.md](GETTING_STARTED.md), [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Look at examples**: [EXAMPLES.md](EXAMPLES.md), [example.py](example.py)
- **Search issues**: https://github.com/shree-crypto/YoutubeViewsPredictor/issues
- **Open a new issue**: Describe your problem with details
- **Read technical docs**: [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)

---

*Last updated: 2024 | Contributors welcome!*
