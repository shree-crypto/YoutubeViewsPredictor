# Training with YouTube Trending Dataset - Quick Guide

This guide walks you through training the YouTube Views Predictor model with real YouTube Trending data from Kaggle.

## Prerequisites

- Python 3.9 or higher installed
- Project dependencies installed (`pip install -r requirements.txt`)
- Downloaded YouTube Trending dataset from Kaggle

## Step 1: Download the Dataset

### Option A: Kaggle Website
1. Go to [Kaggle YouTube Trending Dataset](https://www.kaggle.com/datasnaek/youtube-new)
2. Click "Download" to get the dataset
3. Extract the ZIP file to get CSV files (e.g., `USvideos.csv`, `GBvideos.csv`, etc.)

### Option B: Kaggle API
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API credentials)
kaggle datasets download -d datasnaek/youtube-new
unzip youtube-new.zip -d data/raw/
```

## Step 2: Place Dataset in Project

```bash
# Create raw data directory if it doesn't exist
mkdir -p data/raw

# Move your downloaded dataset
mv ~/Downloads/USvideos.csv data/raw/US_youtube_trending_data.csv
# Or for UK data:
mv ~/Downloads/GBvideos.csv data/raw/GB_youtube_trending_data.csv
```

## Step 3: Train the Model

### Basic Training (All Data)
```bash
python train_model.py --dataset data/raw/US_youtube_trending_data.csv
```

### Training with Sample Size (Faster)
For large datasets (>10k videos), you can use a sample for faster training:
```bash
python train_model.py --dataset data/raw/US_youtube_trending_data.csv --sample-size 5000
```

### Training Output
You should see output like:
```
============================================================
YouTube Views Predictor - Model Training
============================================================

1. Loading YouTube Trending dataset from data/raw/US_youtube_trending_data.csv...
Dataset loaded successfully: 40949 records
View count range: 549 - 225,211,923
Mean views: 2,360,785

2. Extracting features from dataset...
Features extracted: 40949 samples with 25 features

3. Training XGBoost model...

4. Training Results:
------------------------------------------------------------
Training Metrics:
  MAE: 156789.45
  RMSE: 523456.78
  R2: 0.89

Test Metrics:
  MAE: 198234.56
  RMSE: 612345.89
  R2: 0.85

5. Top 10 Most Important Features:
------------------------------------------------------------
  1. is_peak_hour: 0.2345
  2. duration_seconds: 0.1876
  3. title_length: 0.1543
  ...

6. Saving model...

============================================================
Training completed successfully!
Model saved to 'models/' directory
Trained on: YouTube Trending dataset (40949 samples)
============================================================
```

## Step 4: Use the Trained Model

Once training is complete, you can use the model in your applications:

### With Streamlit App
```bash
streamlit run app.py
```

### With API
```bash
python api.py
```

### Programmatically
```python
from utils.model_training import YouTubeViewsPredictor

# Load the trained model
predictor = YouTubeViewsPredictor(model_type='xgboost')
predictor.load_model('models')

# Make predictions
predictions = predictor.predict(features_df)
```

## Dataset Format Details

The data loader automatically handles various YouTube dataset formats:

### Supported Formats
1. **Kaggle Trending YouTube Statistics** - Standard format with columns:
   - `video_id`, `title`, `channel_title`, `category_id`
   - `publish_time`, `tags`, `views`, `likes`, `dislikes`
   - `comment_count`, `description`, `duration` (optional)

2. **Custom YouTube datasets** - Minimum required columns:
   - `title` (required)
   - `views` (required)
   - `duration`, `tags`, `publish_time`, `description` (optional but recommended)

### Automatic Preprocessing
The data loader handles:
- ✅ ISO 8601 duration format (PT15M30S → 930 seconds)
- ✅ Multiple date/time formats
- ✅ Tag separators (pipe | or comma ,)
- ✅ Missing or incomplete data
- ✅ Error video filtering
- ✅ Column name variations (publish_time, publishedAt, publishTime, etc.)

## Troubleshooting

### Issue: FileNotFoundError
**Error:** `Dataset file not found`

**Solution:** Ensure the file path is correct and the file exists:
```bash
ls -la data/raw/US_youtube_trending_data.csv
```

### Issue: Missing Columns
**Error:** `Missing required columns: ['title', 'views']`

**Solution:** Your dataset must have at least `title` and `views` columns. Check column names:
```bash
head -n 1 data/raw/US_youtube_trending_data.csv
```

### Issue: Low Model Performance
**Symptoms:** Low R² score, high error rates

**Solutions:**
1. Use more data (avoid tiny sample sizes)
2. Ensure dataset has duration, tags, and publish_time columns
3. Try different sample sizes
4. Check for data quality issues

### Issue: Memory Error
**Error:** `MemoryError` or system freezes

**Solution:** Use sample size to limit data:
```bash
python train_model.py --dataset data/raw/large_dataset.csv --sample-size 10000
```

## Performance Tips

1. **Optimal Sample Size**: 5,000 - 50,000 videos for good balance
2. **Data Quality**: Remove duplicates, errors, and outliers
3. **Feature Coverage**: Better results with complete metadata (duration, tags, description)
4. **Training Time**: ~1-5 minutes for 10k samples on modern hardware

## Advanced Usage

### Training Multiple Regions
```bash
# Train on US data
python train_model.py --dataset data/raw/USvideos.csv

# Train on UK data
python train_model.py --dataset data/raw/GBvideos.csv

# Compare results
```

### Combining Datasets
```bash
# Combine multiple CSV files
cat data/raw/USvideos.csv > data/raw/combined.csv
tail -n +2 data/raw/GBvideos.csv >> data/raw/combined.csv
tail -n +2 data/raw/CAvideos.csv >> data/raw/combined.csv

# Train on combined data
python train_model.py --dataset data/raw/combined.csv
```

## Next Steps

After training with real data:
1. Compare performance metrics with synthetic data baseline
2. Analyze feature importance for your specific dataset
3. Fine-tune hyperparameters if needed
4. Deploy the model in production
5. Set up periodic retraining with fresh data

## Additional Resources

- [Kaggle YouTube Trending Dataset](https://www.kaggle.com/datasnaek/youtube-new)
- [YouTube Data API Documentation](https://developers.google.com/youtube/v3)
- [Project README](README.md)
- [API Documentation](API_DOCS.md)
