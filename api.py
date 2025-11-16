"""
FastAPI REST API for YouTube Views Predictor

This provides a REST API alternative to the Streamlit interface.

Usage:
    uvicorn api:app --reload
    or
    python api.py
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

from utils.feature_engineering import FeatureExtractor, get_optimal_features
from utils.model_training import YouTubeViewsPredictor
from utils.validators import PredictionRequest, PredictionResponse
from utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(level='INFO')

# Create FastAPI app
app = FastAPI(
    title="YouTube Views Predictor API",
    description="Predict YouTube video views based on video parameters",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[YouTubeViewsPredictor] = None
feature_extractor: Optional[FeatureExtractor] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, feature_extractor
    
    logger.info("Loading model...")
    try:
        predictor = YouTubeViewsPredictor(model_type='xgboost')
        predictor.load_model('models')
        feature_extractor = FeatureExtractor()
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error("Model not found. Please train the model first.")
        predictor = None
        feature_extractor = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "YouTube Views Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    
    return {
        "status": "healthy",
        "model_type": predictor.model_type,
        "model_loaded": predictor.model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_views(request: PredictionRequest):
    """
    Predict video views based on parameters.
    
    Args:
        request: Video parameters for prediction
        
    Returns:
        Prediction response with view count and suggestions
    """
    if predictor is None or feature_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert request to video metadata
        video_metadata = request.to_video_metadata()
        
        # Extract features
        data_dict = {
            'title': video_metadata.title,
            'description': video_metadata.description,
            'tags': video_metadata.tags,
            'duration': video_metadata.duration,
            'publish_time': video_metadata.publish_time
        }
        
        features = feature_extractor.extract_all_features(data_dict)
        
        # Make prediction
        predicted_views = int(predictor.predict(features)[0])
        
        # Calculate confidence interval
        confidence_interval = {
            'lower': int(predicted_views * 0.8),
            'upper': int(predicted_views * 1.2)
        }
        
        # Generate suggestions
        suggestions = generate_suggestions(
            request.title,
            request.duration_minutes,
            request.publish_hour,
            request.tags
        )
        
        # Get feature importance
        feature_importance = None
        if predictor.feature_importance:
            top_features = predictor.get_top_features(n=10)
            feature_importance = dict(top_features) if top_features else None
        
        return PredictionResponse(
            predicted_views=predicted_views,
            confidence_interval=confidence_interval,
            suggestions=suggestions,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/recommendations")
async def get_recommendations():
    """
    Get optimization recommendations.
    
    Returns:
        Dictionary of recommendations by category
    """
    return get_optimal_features()


@app.get("/feature-importance")
async def get_feature_importance(top_n: int = 15):
    """
    Get feature importance scores.
    
    Args:
        top_n: Number of top features to return
        
    Returns:
        List of top features with importance scores
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if predictor.feature_importance is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance not available"
        )
    
    top_features = predictor.get_top_features(n=top_n)
    
    return {
        "features": [
            {"name": name, "importance": float(importance)}
            for name, importance in top_features
        ]
    }


@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model metadata and configuration
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": predictor.model_type,
        "feature_count": len(predictor.feature_names) if predictor.feature_names else 0,
        "features": predictor.feature_names,
        "has_feature_importance": predictor.feature_importance is not None
    }


def generate_suggestions(
    title: str,
    duration_minutes: float,
    publish_hour: int,
    tags: Optional[str]
) -> List[str]:
    """
    Generate optimization suggestions based on input parameters.
    
    Args:
        title: Video title
        duration_minutes: Video duration in minutes
        publish_hour: Publishing hour (0-23)
        tags: Comma-separated tags
        
    Returns:
        List of suggestion strings
    """
    suggestions = []
    
    # Title length
    title_len = len(title)
    if title_len < 50:
        suggestions.append(f"❌ Title too short ({title_len} chars). Optimal: 50-70 characters.")
    elif title_len > 70:
        suggestions.append(f"⚠️ Title too long ({title_len} chars). Optimal: 50-70 characters.")
    else:
        suggestions.append(f"✅ Title length is optimal ({title_len} chars).")
    
    # Publish hour
    if 18 <= publish_hour <= 21:
        suggestions.append(f"✅ Publishing during peak hours ({publish_hour}:00).")
    else:
        suggestions.append(f"⚠️ Non-peak hour ({publish_hour}:00). Consider 6-9 PM for better reach.")
    
    # Duration
    if 7 <= duration_minutes <= 15:
        suggestions.append(f"✅ Duration is optimal ({duration_minutes:.1f} min).")
    elif duration_minutes < 7:
        suggestions.append(f"⚠️ Video might be too short ({duration_minutes:.1f} min). Consider 7-15 min.")
    else:
        suggestions.append(f"ℹ️ Longer video ({duration_minutes:.1f} min). Works for in-depth content.")
    
    # Tags
    if tags:
        tag_count = len([t for t in tags.split(',') if t.strip()])
        if tag_count < 10:
            suggestions.append(f"⚠️ Add more tags ({tag_count} tags). Aim for 10-15 relevant tags.")
        else:
            suggestions.append(f"✅ Good tag count ({tag_count} tags).")
    else:
        suggestions.append("❌ No tags provided. Add 10-15 relevant tags.")
    
    return suggestions


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
