# Getting Started with YouTube Views Predictor

Welcome! This guide will walk you through everything you need to know to start using the YouTube Views Predictor, from installation to making your first prediction.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First Prediction](#your-first-prediction)
4. [Understanding the Results](#understanding-the-results)
5. [Using the Web Interface](#using-the-web-interface)
6. [Using Programmatically](#using-programmatically)
7. [Next Steps](#next-steps)

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.8 or higher** installed on your system
  - Check with: `python --version` or `python3 --version`
  - Download from: https://www.python.org/downloads/
  
- **pip** package manager (usually comes with Python)
  - Check with: `pip --version` or `pip3 --version`
  
- **Git** (to clone the repository)
  - Check with: `git --version`
  - Download from: https://git-scm.com/downloads

- **2GB free disk space** for dependencies and models

- **Internet connection** for initial setup

---

## Installation

### Step 1: Clone the Repository

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/shree-crypto/YoutubeViewsPredictor.git
cd YoutubeViewsPredictor
```

This creates a `YoutubeViewsPredictor` directory with all project files.

### Step 2: Create a Virtual Environment (Recommended)

A virtual environment keeps this project's dependencies separate from other Python projects.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Note:** If you encounter an error with `torch`, you can skip it as it's optional for basic functionality:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm streamlit matplotlib seaborn plotly nltk textblob wordcloud joblib python-dotenv
```

This will take a few minutes. You'll see progress bars as packages are installed.

### Step 4: Verify Installation

Let's verify everything is installed correctly:

```bash
python -c "import pandas, sklearn, xgboost, streamlit; print('✓ All packages installed successfully!')"
```

If you see the success message, you're ready to proceed!

---

## Your First Prediction

### Step 1: Train the Model

Before making predictions, you need to train the machine learning model:

```bash
python train_model.py
```

**What you'll see:**

```
============================================================
YouTube Views Predictor - Model Training
============================================================

1. Creating sample dataset...
Dataset shape: (1000, 26)
View count range: 100 - 165320
Mean views: 47195

2. Preparing features and target...
Number of features: 25

3. Training XGBoost model...

4. Training Results:
------------------------------------------------------------
Training Metrics:
  MAE: 470.95
  RMSE: 680.77
  R2: 1.00

Test Metrics:
  MAE: 16901.24
  RMSE: 21005.05
  R2: 0.69

5. Top 10 Most Important Features:
------------------------------------------------------------
  1. is_peak_hour: 0.6204
  2. is_weekend: 0.1258
  3. has_question_mark: 0.0531
  4. duration_minutes: 0.0462
  5. tags_count: 0.0182
  ...

6. Saving model...
Model saved to models

============================================================
Training completed successfully!
Model saved to 'models/' directory
============================================================
```

**What just happened?**
- Generated 1,000 sample YouTube videos with realistic parameters
- Extracted 25+ features from each video (title length, publish time, duration, etc.)
- Trained an XGBoost machine learning model to predict views
- Evaluated the model's accuracy (R² of 0.69 means it explains 69% of variance)
- Saved the trained model to the `models/` directory

**Note:** This uses synthetic data for demonstration. For real predictions, you'd train on actual YouTube data.

### Step 2: Launch the Web Application

Now start the interactive web interface:

```bash
streamlit run app.py
```

**What you'll see:**

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

Your default web browser should automatically open to http://localhost:8501

**If the browser doesn't open automatically:**
- Manually open your browser and go to: http://localhost:8501
- Or try the Network URL shown in the terminal

### Step 3: Make Your First Prediction

The web app has three tabs. Let's start with the **Prediction** tab:

#### Fill in Video Details:

1. **Video Title**: "How to Build Amazing Projects in 2024!"
   
2. **Description**: "Learn step-by-step how to build impressive projects that will boost your skills and portfolio."
   
3. **Tags**: "tutorial,programming,projects,beginners,2024"
   
4. **Duration**: 10.0 (minutes)
   
5. **Publish Date**: Today's date
   
6. **Publish Time**: 19:00 (7 PM)
   
7. **Category**: Education

#### Options:
- ✓ Check "Show confidence interval"
- ✓ Check "Compare with optimal parameters"

#### Click "Predict Views" 🎯

**What you'll see:**

```
📊 Prediction Results

Predicted Views: 52,847 views
Confidence Interval: 42,278 - 63,417 views

🎯 Optimization Suggestions

Your Parameters:
• Title length: 42 characters
• Publishing: Friday at 7:00 PM (peak hour ✓)
• Duration: 10.0 minutes
• Tags: 5 tags

Recommendations:
✓ Publishing time is optimal (peak hours)
✓ Weekend timing is good
• Consider lengthening title to 50-70 characters
• Add more tags (aim for 10-15)
• Add question mark or exclamation to title

Potential Improvement: +15.3% views
```

Congratulations! You've made your first prediction! 🎉

---

## Understanding the Results

### Predicted Views
The main number (e.g., "52,847 views") is the model's best estimate based on your video parameters.

### Confidence Interval
The range (e.g., "42,278 - 63,417") shows the expected range of views:
- **Lower bound (80%)**: Conservative estimate
- **Upper bound (120%)**: Optimistic estimate
- The actual result could fall anywhere in this range

### Optimization Suggestions
These are data-driven recommendations to improve your predicted views:
- **Green checkmarks (✓)**: Parameters that are already optimal
- **Bullet points**: Areas for improvement
- **Potential Improvement**: Estimated increase if you follow all suggestions

### Feature Importance (Tab 2)
Shows which factors most influence view predictions:
1. **Peak hour publishing** (6-9 PM): Most important factor
2. **Weekend uploads**: Significantly impacts views
3. **Title question marks**: Increases engagement
4. **Video duration**: Optimal length matters
5. **Tag count**: More tags = better discoverability

---

## Using the Web Interface

### Tab 1: Prediction 🎯

**Input Fields Explained:**

- **Title**: Your video's title (50-70 characters optimal)
  - ✅ Good: "How to Learn Python in 10 Minutes!"
  - ❌ Avoid: "video" (too short, not descriptive)

- **Description**: Detailed description of your video
  - Aim for 200-300 words
  - Include relevant keywords
  - Add 2-3 links if relevant

- **Tags**: Comma-separated keywords
  - Example: "python,tutorial,programming,beginners,coding"
  - Use 10-15 tags
  - Mix broad and specific terms

- **Duration**: Video length in minutes
  - Quick content: 3-5 minutes
  - Tutorials: 7-15 minutes
  - Deep dives: 15-30 minutes

- **Publish Date/Time**: When you'll upload
  - Best times: 6-9 PM (18:00-21:00)
  - Best days: Friday-Sunday
  - Avoid: Early mornings Monday-Wednesday

- **Category**: Choose the most relevant category
  - Affects algorithm recommendations
  - Be accurate for best results

### Tab 2: Feature Importance 📊

This tab shows a bar chart of the top 15 features that influence view predictions.

**How to use it:**
- Identify which factors matter most (taller bars)
- Focus optimization efforts on high-importance features
- Understand why your predictions are what they are

### Tab 3: Recommendations 💡

Evidence-based best practices for maximizing views:

**Title Optimization:**
- Keep between 50-70 characters
- Include 1-2 numbers
- Add "?" or "!" for engagement
- Use emotional/power words
- Capitalize 1-2 key words (not entire title)

**Publishing Time:**
- Upload 6-9 PM local time
- Friday-Sunday preferred
- Consider your audience's timezone

**Duration:**
- Match content type to length
- Aim for 7-15 minutes for most content
- Maintain high watch time percentage

**Metadata:**
- Use 10-15 relevant tags
- Write comprehensive descriptions
- Include timestamps for longer videos
- Add 2-3 relevant links

---

## Using Programmatically

If you want to integrate predictions into your own scripts or applications:

### Basic Example

```python
from utils.model_training import YouTubeViewsPredictor
from utils.feature_engineering import FeatureExtractor

# Load the trained model
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

# Make prediction
predicted_views = predictor.predict(features)
print(f"Predicted views: {predicted_views[0]:,.0f}")
```

**Output:**
```
Predicted views: 52,847
```

### Batch Predictions

```python
import pandas as pd

# Load multiple videos from CSV
videos_df = pd.read_csv('my_videos.csv')

# Extract features for all videos
features_df = extractor.process_dataframe(videos_df)

# Predict all at once
predictions = predictor.predict(features_df)

# Add to dataframe
videos_df['predicted_views'] = predictions
videos_df.to_csv('videos_with_predictions.csv', index=False)
```

### Get Top Features

```python
# See which features are most important
top_features = predictor.get_top_features(n=10)

for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
```

**Output:**
```
is_peak_hour: 0.6204
is_weekend: 0.1258
has_question_mark: 0.0531
duration_minutes: 0.0462
tags_count: 0.0182
...
```

---

## Next Steps

### 1. Train with Real Data

The sample model uses synthetic data. For accurate predictions:

**Option A: Use YouTube API**
```python
# Get YouTube API key from Google Cloud Console
# Use YouTube Data API v3 to fetch video statistics
# Save to CSV and retrain model
```

**Option B: Use Kaggle Datasets**
- [Trending YouTube Video Statistics](https://www.kaggle.com/datasnaek/youtube-new)
- [YouTube Video Dataset](https://www.kaggle.com/rsrishav/youtube-trending-video-dataset)

**Retrain with your data:**
```python
import pandas as pd
from utils.feature_engineering import FeatureExtractor
from utils.model_training import YouTubeViewsPredictor

# Load your data
df = pd.read_csv('your_youtube_data.csv')

# Extract features
extractor = FeatureExtractor()
features_df = extractor.process_dataframe(df)

# Train model
predictor = YouTubeViewsPredictor(model_type='xgboost')
results = predictor.train(features_df, df['views'])

# Save model
predictor.save_model('models')
```

### 2. Explore the Notebooks

Check out `notebooks/analysis_example.ipynb` for:
- Data exploration examples
- Feature importance visualization
- Model performance analysis
- Custom analysis templates

### 3. Experiment with Different Videos

Try different scenarios:
- Short vs. long videos
- Different publishing times
- Various title styles
- Multiple categories

Compare predictions to understand what works best!

### 4. Track Your Real Results

After publishing videos:
1. Compare predicted views with actual views
2. Note which predictions were accurate
3. Identify patterns in prediction errors
4. Adjust your video strategy accordingly

### 5. Customize the Model

Advanced users can:
- Add custom features in `utils/feature_engineering.py`
- Tune hyperparameters in `utils/model_training.py`
- Experiment with different model types (LightGBM, etc.)
- Implement deep learning approaches

---

## Troubleshooting

### Model not found error
**Error:** "Model not found! Please run train_model.py first."

**Solution:**
```bash
python train_model.py
```

### Module not found errors
**Error:** "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
pip install -r requirements.txt
# Or install packages individually
pip install pandas numpy scikit-learn xgboost streamlit
```

### NLTK data missing
**Error:** "Resource stopwords not found"

**Solution:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Streamlit port already in use
**Error:** "Address already in use"

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or stop the existing process
# On Linux/Mac:
lsof -ti:8501 | xargs kill
# On Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Low prediction accuracy
**Issue:** Predictions don't match actual results

**Solutions:**
1. Train on real YouTube data (not synthetic)
2. Use a larger dataset (1000+ videos)
3. Include recent videos (last 6-12 months)
4. Retrain the model periodically

### Application runs slowly
**Issue:** Slow predictions or training

**Solutions:**
1. Use LightGBM instead of XGBoost (faster):
   ```python
   predictor = YouTubeViewsPredictor(model_type='lightgbm')
   ```
2. Reduce the dataset size for training
3. Use fewer features (remove low-importance ones)

---

## Getting Help

**Found a bug or have a question?**

1. **Check existing documentation:**
   - [README.md](README.md) - Project overview
   - [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage instructions
   - [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) - Technical details

2. **Search existing issues:**
   - https://github.com/shree-crypto/YoutubeViewsPredictor/issues

3. **Create a new issue:**
   - Describe your problem
   - Include error messages
   - Show what you tried
   - Include system information (OS, Python version)

4. **Community support:**
   - Share your use case
   - Ask questions
   - Contribute improvements

---

## Quick Reference

### Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run web interface
streamlit run app.py

# Run tests
python test_basic.py

# Deactivate virtual environment
deactivate

# Reactivate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Project Structure

```
YoutubeViewsPredictor/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── requirements.txt        # Dependencies
├── README.md              # Project overview
├── GETTING_STARTED.md     # This file
├── USAGE_GUIDE.md         # Detailed usage guide
├── TECHNICAL_DOCS.md      # Technical documentation
├── utils/
│   ├── feature_engineering.py  # Feature extraction
│   └── model_training.py       # ML models
├── models/                # Trained models (created after training)
├── data/                  # Data files (created after training)
└── notebooks/            # Jupyter notebooks for analysis
```

---

## What's Next?

You're now ready to:
- ✅ Make predictions for your videos
- ✅ Understand what drives YouTube views
- ✅ Optimize your video parameters
- ✅ Integrate predictions into your workflow

**Happy predicting! 📺 🚀**

For more advanced usage, check out:
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Comprehensive usage instructions
- [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) - Technical implementation details

---

*Made with ❤️ for YouTube creators seeking data-driven insights*
