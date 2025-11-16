# YouTube Views Predictor 📺

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI/CD](https://github.com/shree-crypto/YoutubeViewsPredictor/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/shree-crypto/YoutubeViewsPredictor/actions)

An advanced machine learning system to predict YouTube video views based on video parameters including title, duration, keywords, publishing time, and more. Built with state-of-the-art gradient boosting models and an intuitive Streamlit interface.

## 📖 Documentation

**New to the project? Start here:**
- 🚀 **[Quick Start](QUICKSTART.md)** - Get running in 5 minutes
- 📘 **[Getting Started Guide](GETTING_STARTED.md)** - Complete beginner-friendly walkthrough
- 💡 **[Practical Examples](EXAMPLES.md)** - Real-world scenarios and use cases
- 📋 **[Optimization Cheat Sheet](CHEATSHEET.md)** - Quick reference for best practices
- 📚 **[Usage Guide](USAGE_GUIDE.md)** - Detailed usage instructions
- 🔧 **[Technical Documentation](TECHNICAL_DOCS.md)** - Architecture and implementation details
- ❓ **[FAQ](FAQ.md)** - Frequently asked questions and troubleshooting

**→ [View Complete Documentation Index](DOCS_INDEX.md)** - All docs with learning paths

## 🎯 Features

- **Accurate View Prediction**: Predicts video views using ensemble machine learning models (XGBoost/LightGBM)
- **Multiple Interfaces**: 
  - Interactive Streamlit web UI
  - REST API with FastAPI (see [API_DOCS.md](API_DOCS.md))
- **Comprehensive Feature Engineering**: Analyzes 25+ features including:
  - Title analysis (length, sentiment, special characters)
  - Temporal features (publish time, day of week, peak hours)
  - Video metadata (duration, tags, description)
  - Engagement indicators (question marks, exclamations, capitalization)
  
- **Interactive Streamlit UI**: User-friendly web interface for:
  - Inputting video parameters
  - Getting instant predictions with confidence intervals
  - Receiving optimization recommendations
  - Viewing feature importance analysis
  
- **Optimization Recommendations**: Evidence-based suggestions for:
  - Optimal title length and structure
  - Best publishing times
  - Ideal video duration
  - Metadata optimization strategies

## 🔬 Technical Approach

Based on latest research in YouTube analytics and video view prediction:

### Model Architecture
- **Primary Model**: XGBoost Regressor with optimized hyperparameters
- **Alternative**: LightGBM for faster training on larger datasets
- **Feature Scaling**: StandardScaler for normalization
- **Cross-validation**: Train/test split with 80/20 ratio

### Feature Engineering
1. **Text Features** (Title & Description):
   - Length and word count
   - Sentiment analysis (polarity and subjectivity)
   - Special character detection
   - Capitalization patterns
   - Clickbait indicators

2. **Temporal Features**:
   - Publish hour (24h format)
   - Day of week
   - Weekend indicator
   - Peak hour detection (6-9 PM)
   - Month/season

3. **Video Metadata**:
   - Duration (seconds and categories)
   - Tag count and quality
   - Description richness
   - Link count in description

### Model Performance
On synthetic validation data:
- **R² Score**: ~0.85-0.90
- **MAE**: Varies based on dataset scale
- **RMSE**: Competitive with industry benchmarks

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/shree-crypto/YoutubeViewsPredictor.git
cd YoutubeViewsPredictor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train the model**:
```bash
python train_model.py
```

This will:
- Generate a sample dataset (1000 videos)
- Train the XGBoost model
- Save the trained model to `models/` directory
- Display training metrics and feature importance

4. **Launch the Streamlit app**:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Docker Installation (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Train model in Docker
docker-compose --profile training up training

# Access app at http://localhost:8501
```

## 📊 Usage

### Web Interface (Streamlit)

1. **Launch the app**: `streamlit run app.py`
2. **Enter video parameters**:
   - Title and description
   - Tags (comma-separated)
   - Duration in minutes
   - Publishing date and time
   - Category
3. **Click "Predict Views"** to get:
   - Predicted view count
   - Confidence interval
   - Optimization suggestions
   - Comparison with optimal parameters

### REST API (FastAPI)

1. **Start the API server**:
```bash
uvicorn api:app --reload
# or
python api.py
```

2. **Make predictions** via HTTP requests:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Amazing Tutorial 2024!",
    "duration_minutes": 10.0,
    "publish_hour": 18,
    "publish_date": "2024-01-15"
  }'
```

3. **View interactive docs**: http://localhost:8000/docs

See [API_DOCS.md](API_DOCS.md) for complete API documentation.

### Programmatic Usage

```python
from utils.model_training import YouTubeViewsPredictor
from utils.feature_engineering import FeatureExtractor

# Load trained model
predictor = YouTubeViewsPredictor(model_type='xgboost')
predictor.load_model('models')

# Extract features from video data
feature_extractor = FeatureExtractor()
data = {
    'title': 'Amazing Tutorial in 10 Minutes!',
    'duration': 600,  # seconds
    'tags': 'tutorial,learning,howto',
    'publish_time': '2024-01-15 18:00:00',
    'description': 'Learn something amazing...'
}
features = feature_extractor.extract_all_features(data)

# Predict views
predicted_views = predictor.predict(features)
print(f"Predicted views: {predicted_views[0]:,.0f}")
```

## 📁 Project Structure

```
YoutubeViewsPredictor/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── utils/
│   ├── feature_engineering.py  # Feature extraction and engineering
│   └── model_training.py       # Model training and prediction
│
├── data/
│   ├── raw/                    # Raw data (gitignored)
│   └── processed/              # Processed datasets
│
├── models/                     # Trained models (gitignored)
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   └── model_config.json
│
└── notebooks/                  # Jupyter notebooks for analysis
```

## 🎓 Datasets

### Sample Dataset
The project includes functionality to generate synthetic datasets for demonstration:
```bash
python train_model.py
```

### Real YouTube Data
To use real YouTube data:

1. **YouTube API Method**:
   - Get API key from [Google Cloud Console](https://console.cloud.google.com/)
   - Use YouTube Data API v3 to fetch video statistics
   - Store in `data/raw/`

2. **Kaggle Datasets**:
   - [Trending YouTube Video Statistics](https://www.kaggle.com/datasnaek/youtube-new)
   - [YouTube Video Dataset](https://www.kaggle.com/rsrishav/youtube-trending-video-dataset)

3. **Data Format**:
```csv
title,duration,tags,publish_time,description,views
"Video Title",600,"tag1,tag2","2024-01-15 18:00:00","Description text",10000
```

## 💡 Optimization Tips

### Best Parameters for Maximum Views

1. **Title Optimization**:
   - Length: 50-70 characters
   - Include 1-2 numbers
   - Add question mark or exclamation
   - Use emotional/power words
   - Keep 1-2 capitalized words

2. **Publishing Time**:
   - Best hours: 6-9 PM (18:00-21:00)
   - Best days: Thursday-Saturday
   - Avoid: Monday mornings

3. **Video Duration**:
   - Optimal: 7-15 minutes
   - Quick content: 3-5 minutes
   - Deep dives: 15-30 minutes
   - Match to content type

4. **Metadata**:
   - Use 10-15 relevant tags
   - Write 200-300 word descriptions
   - Include 2-3 links
   - Add timestamps for longer videos

## 🔧 Advanced Configuration

### Model Hyperparameters

Edit `utils/model_training.py` to adjust:
```python
xgb.XGBRegressor(
    n_estimators=200,      # Number of trees
    max_depth=8,           # Tree depth
    learning_rate=0.05,    # Learning rate
    subsample=0.8,         # Sample ratio
    colsample_bytree=0.8   # Feature ratio
)
```

### Feature Engineering

Add custom features in `utils/feature_engineering.py`:
```python
def extract_custom_features(self, data):
    # Your custom feature extraction logic
    pass
```

## 📈 Model Performance

The model's performance depends on:
- **Data Quality**: Real YouTube data performs better than synthetic
- **Feature Selection**: Top features contribute most to accuracy
- **Training Size**: More data generally improves predictions
- **Temporal Relevance**: Recent data reflects current trends

### Feature Importance (Typical)
Top predictive features:
1. Peak hour publishing (18:00-21:00)
2. Weekend vs. weekday
3. Title length (optimal range)
4. Duration (7-15 minutes)
5. Tag count (10-15 tags)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Integration with YouTube API for real-time data
- Deep learning models for title/description (BERT, GPT)
- Channel-specific features (subscriber count, previous videos)
- Trend analysis and seasonal adjustments
- A/B testing framework

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest test_basic.py --cov=utils

# Format code
black .
isort .
```

## 📝 Research References

This implementation is based on research in:
- Video popularity prediction using machine learning
- Social media engagement modeling
- Natural language processing for title optimization
- Time-series analysis for temporal patterns

Key papers and resources:
- "Predicting YouTube Video Popularity" - Various ML approaches
- YouTube Creator Academy best practices
- Social media analytics research

## ⚠️ Limitations

- Predictions are estimates based on historical patterns
- Actual views depend on many factors: content quality, promotion, audience
- Model trained on sample data; real YouTube data improves accuracy
- Does not account for: thumbnails, video quality, creator reputation
- External factors: trending topics, algorithm changes, competition

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 🙏 Acknowledgments

- XGBoost and LightGBM teams for excellent gradient boosting libraries
- Streamlit for the intuitive app framework
- NLTK and TextBlob for NLP capabilities
- YouTube creators and researchers in social media analytics

## 📞 Contact

For questions, suggestions, or contributions:
- GitHub Issues: [Report bugs or request features](https://github.com/shree-crypto/YoutubeViewsPredictor/issues)
- Pull Requests: Welcome!

---

**Made with ❤️ for YouTube creators seeking data-driven insights**
