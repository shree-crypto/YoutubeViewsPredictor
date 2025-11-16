# Getting Started with Development

This guide will help you set up your development environment and start working on the YouTube Views Predictor project.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Git
- pip (Python package manager)
- (Optional) Docker and Docker Compose

## Step 1: Environment Setup

### 1.1 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 1.2 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## Step 2: YouTube API Setup

### 2.1 Get API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API Key)
5. Copy the API key

### 2.2 Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# Set YOUTUBE_API_KEY=your_actual_api_key_here
```

## Step 3: Create Project Structure

```bash
# Create necessary directories
mkdir -p data/raw data/processed data/features data/models
mkdir -p src/data src/preprocessing src/models src/training src/deployment
mkdir -p notebooks tests scripts docs configs logs

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/preprocessing/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/deployment/__init__.py
touch tests/__init__.py

# Create placeholder files for data directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/features/.gitkeep
touch data/models/.gitkeep
```

## Step 4: Verify Installation

```bash
# Test Python imports
python -c "import pandas; import numpy; import sklearn; print('All packages imported successfully!')"

# Check Python version
python --version

# List installed packages
pip list
```

## Step 5: Start with Phase 1 (Data Collection)

### 5.1 Create Your First Module

Create `src/data/data_collector.py`:

```python
"""
YouTube Data Collector
Fetches video data using YouTube Data API v3
"""

import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

class YouTubeDataCollector:
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
    
    def get_video_stats(self, video_id):
        """Fetch statistics for a single video"""
        request = self.youtube.videos().list(
            part='statistics,snippet,contentDetails',
            id=video_id
        )
        response = request.execute()
        return response
    
    def search_videos(self, query, max_results=50):
        """Search for videos by query"""
        request = self.youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            maxResults=max_results
        )
        response = request.execute()
        return response

if __name__ == '__main__':
    collector = YouTubeDataCollector()
    print("Data collector initialized successfully!")
```

### 5.2 Create Your First Notebook

Create `notebooks/01_data_exploration.ipynb`:
- Start exploring the YouTube API
- Test data collection
- Analyze initial data samples
- Document findings

## Step 6: Development Workflow

### Daily Workflow
1. Activate virtual environment
2. Pull latest changes: `git pull`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Test your changes
6. Commit and push
7. Create a pull request

### Code Quality
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Run tests
pytest tests/
```

## Step 7: Running with Docker (Optional)

```bash
# Build the Docker image
docker-compose build

# Run the application
docker-compose up

# Stop the application
docker-compose down
```

## Common Issues and Solutions

### Issue: API Key Not Working
- Verify the API key is correct in `.env`
- Check that YouTube Data API v3 is enabled in Google Cloud Console
- Ensure you haven't exceeded your quota

### Issue: Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version`

### Issue: Permission Errors
- On Linux/Mac, you may need to use `chmod +x` on scripts
- Ensure you have write permissions in the project directory

## Useful Commands

```bash
# Check disk space in data directories
du -sh data/*

# Count Python files
find src -name "*.py" | wc -l

# Run specific test file
pytest tests/test_data_collector.py -v

# Generate coverage report
pytest --cov=src tests/

# Start Jupyter notebook
jupyter notebook notebooks/
```

## Resources

- [YouTube Data API Documentation](https://developers.google.com/youtube/v3/docs)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

## Getting Help

- Check [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) for architecture details
- Check [ROADMAP.md](ROADMAP.md) for development phases
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- Open an issue on GitHub for questions

## Next Steps

1. ✅ Complete environment setup
2. ✅ Get YouTube API key
3. ✅ Create project structure
4. → Start data collection (Phase 1, Week 1)
5. → Begin exploratory data analysis

---

Happy coding! 🚀
