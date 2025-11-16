# Quick Start - YouTube Views Predictor

Get up and running in 5 minutes! ⚡

## 1. Install (2 minutes)

```bash
# Clone repository
git clone https://github.com/shree-crypto/YoutubeViewsPredictor.git
cd YoutubeViewsPredictor

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm streamlit plotly nltk textblob joblib
```

## 2. Train Model (1 minute)

```bash
python train_model.py
```

**Expected output:**
```
Training completed successfully!
Model saved to 'models/' directory
```

## 3. Launch App (30 seconds)

```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

## 4. Make Prediction (1 minute)

In the web app:

1. **Title:** "How to Learn Python in 10 Minutes!"
2. **Description:** "Complete beginner's guide to Python programming"
3. **Tags:** "python,tutorial,programming,beginners"
4. **Duration:** 10.0 minutes
5. **Time:** 7:00 PM (19:00)
6. **Date:** Friday or Saturday
7. Click **"Predict Views"** 🎯

**Result:** View prediction with optimization tips! 📊

---

## Quick Tips 💡

### Best Parameters for Maximum Views:
- **Title:** 50-70 characters with "?" or "!"
- **Time:** 6-9 PM (18:00-21:00)
- **Days:** Friday-Sunday
- **Duration:** 7-15 minutes
- **Tags:** 10-15 relevant tags

### Try These Example Videos:

**Example 1: Tutorial**
```
Title: "Learn JavaScript in 15 Minutes! 🚀"
Duration: 15 min
Time: 7:00 PM Friday
Tags: javascript,tutorial,programming,web,beginners
```

**Example 2: Review**
```
Title: "Is This the Best Budget Laptop in 2024?"
Duration: 8 min
Time: 6:00 PM Saturday
Tags: laptop,review,tech,budget,2024
```

**Example 3: How-To**
```
Title: "How I Built This Amazing App"
Duration: 12 min
Time: 8:00 PM Sunday
Tags: app,development,coding,project,tutorial
```

---

## Programmatic Usage

```python
from utils.model_training import YouTubeViewsPredictor
from utils.feature_engineering import FeatureExtractor

# Load model
predictor = YouTubeViewsPredictor(model_type='xgboost')
predictor.load_model('models')

# Create feature extractor
extractor = FeatureExtractor()

# Your video
video = {
    'title': 'Amazing Tutorial!',
    'duration': 600,
    'tags': 'tutorial,programming',
    'publish_time': '2024-01-15 19:00:00',
    'description': 'Learn something cool...'
}

# Predict
features = extractor.extract_all_features(video)
views = predictor.predict(features)
print(f"Predicted: {views[0]:,.0f} views")
```

---

## Troubleshooting

**Model not found?**
```bash
python train_model.py
```

**Import errors?**
```bash
pip install -r requirements.txt
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

---

## What's Next?

- 📚 Read [GETTING_STARTED.md](GETTING_STARTED.md) for detailed walkthrough
- 📖 Check [USAGE_GUIDE.md](USAGE_GUIDE.md) for advanced features
- 🔧 See [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) for implementation details
- 📊 Explore `notebooks/` for analysis examples

---

**Ready to predict! 🎉**

*Questions? Open an issue: https://github.com/shree-crypto/YoutubeViewsPredictor/issues*
