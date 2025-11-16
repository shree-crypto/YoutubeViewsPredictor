# REST API Documentation

## YouTube Views Predictor API

A FastAPI-based REST API for predicting YouTube video views.

### Base URL

```
http://localhost:8000
```

### Quick Start

1. **Train the model** (if not already trained):
   ```bash
   python train_model.py
   ```

2. **Start the API server**:
   ```bash
   uvicorn api:app --reload
   # or
   python api.py
   ```

3. **Access the interactive documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

---

## Endpoints

### 1. Root
Get API information.

**GET** `/`

**Response:**
```json
{
  "message": "YouTube Views Predictor API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### 2. Health Check
Check if the API and model are loaded and ready.

**GET** `/health`

**Response (Healthy):**
```json
{
  "status": "healthy",
  "model_type": "xgboost",
  "model_loaded": true
}
```

**Response (Unhealthy):**
```json
{
  "status": "unhealthy",
  "message": "Model not loaded"
}
```

---

### 3. Predict Views
Predict video views based on parameters.

**POST** `/predict`

**Request Body:**
```json
{
  "title": "Amazing Python Tutorial 2024!",
  "description": "Learn Python from scratch...",
  "tags": "python,tutorial,programming,coding,2024",
  "duration_minutes": 10.0,
  "publish_hour": 18,
  "publish_date": "2024-01-15",
  "category": "Education"
}
```

**Parameters:**
- `title` (string, required): Video title (1-200 characters)
- `description` (string, optional): Video description (max 5000 characters)
- `tags` (string, optional): Comma-separated tags (max 500 characters)
- `duration_minutes` (float, required): Video duration in minutes (0-1440)
- `publish_hour` (int, required): Publishing hour in 24h format (0-23)
- `publish_date` (string, required): Publishing date (YYYY-MM-DD format)
- `category` (string, optional): Video category (default: "Education")

**Response:**
```json
{
  "predicted_views": 125000,
  "confidence_interval": {
    "lower": 100000,
    "upper": 150000
  },
  "suggestions": [
    "✅ Title length is optimal (32 chars).",
    "✅ Publishing during peak hours (18:00).",
    "✅ Duration is optimal (10.0 min).",
    "✅ Good tag count (5 tags)."
  ],
  "feature_importance": {
    "is_peak_hour": 0.15,
    "title_length": 0.12,
    "duration_minutes": 0.10
  }
}
```

---

### 4. Get Recommendations
Get optimization recommendations for video parameters.

**GET** `/recommendations`

**Response:**
```json
{
  "title_recommendations": [
    "Keep title length between 50-70 characters",
    "Use 1-2 capitalized words for emphasis",
    "Include numbers or statistics"
  ],
  "temporal_recommendations": [
    "Upload during peak hours (6-9 PM local time)",
    "Friday and weekend uploads tend to perform better"
  ],
  "duration_recommendations": [
    "Optimal duration: 7-15 minutes for most content",
    "Shorter videos (< 5 min) for quick tips/news"
  ],
  "metadata_recommendations": [
    "Use 10-15 relevant tags",
    "Write detailed descriptions (200-300 words)"
  ]
}
```

---

### 5. Get Feature Importance
Get the most important features for predictions.

**GET** `/feature-importance?top_n=10`

**Query Parameters:**
- `top_n` (int, optional): Number of top features to return (default: 15)

**Response:**
```json
{
  "features": [
    {"name": "is_peak_hour", "importance": 0.15},
    {"name": "title_length", "importance": 0.12},
    {"name": "duration_minutes", "importance": 0.10},
    {"name": "tags_count", "importance": 0.08}
  ]
}
```

---

### 6. Get Model Info
Get information about the loaded model.

**GET** `/model-info`

**Response:**
```json
{
  "model_type": "xgboost",
  "feature_count": 25,
  "features": [
    "title_length",
    "title_word_count",
    "publish_hour",
    "duration_seconds"
  ],
  "has_feature_importance": true
}
```

---

## Usage Examples

### Python (requests)

```python
import requests

# Predict views
url = "http://localhost:8000/predict"
data = {
    "title": "Complete Python Tutorial 2024!",
    "description": "Learn Python from scratch in this comprehensive tutorial.",
    "tags": "python,tutorial,programming,coding,beginners",
    "duration_minutes": 12.0,
    "publish_hour": 19,
    "publish_date": "2024-01-15",
    "category": "Education"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Predicted views: {result['predicted_views']:,}")
print(f"Suggestions: {result['suggestions']}")
```

### cURL

```bash
# Predict views
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Amazing Python Tutorial 2024!",
    "duration_minutes": 10.0,
    "publish_hour": 18,
    "publish_date": "2024-01-15",
    "tags": "python,tutorial,coding"
  }'

# Get recommendations
curl "http://localhost:8000/recommendations"

# Check health
curl "http://localhost:8000/health"
```

### JavaScript (fetch)

```javascript
// Predict views
const predictViews = async () => {
  const data = {
    title: "Amazing Python Tutorial 2024!",
    duration_minutes: 10.0,
    publish_hour: 18,
    publish_date: "2024-01-15",
    tags: "python,tutorial,coding"
  };

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  console.log('Predicted views:', result.predicted_views);
};
```

---

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid input data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Model not loaded

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Validation Rules

- **Title**: 1-200 characters, cannot be empty
- **Description**: Max 5000 characters
- **Tags**: Max 500 characters
- **Duration**: 0.5-1440 minutes (30 seconds to 24 hours)
- **Publish Hour**: 0-23 (24-hour format)
- **Publish Date**: YYYY-MM-DD format

---

## Production Deployment

### Docker

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Environment Variables

Create a `.env` file:

```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_DIR=models
LOG_LEVEL=INFO
```

---

## Rate Limiting (Recommended for Production)

Consider adding rate limiting for production:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_views(request: Request, pred_request: PredictionRequest):
    # ... prediction logic
```

---

## Testing the API

```bash
# Run tests
pytest test_api.py -v

# Test with coverage
pytest test_api.py --cov=api --cov-report=html
```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/shree-crypto/YoutubeViewsPredictor/issues
- Documentation: See project README.md
