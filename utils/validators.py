"""
Data Validation Module

Validates input data using Pydantic models for type safety and validation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from datetime import datetime


class VideoMetadata(BaseModel):
    """Video metadata model with validation."""
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Video title"
    )
    description: Optional[str] = Field(
        default="",
        max_length=5000,
        description="Video description"
    )
    tags: Optional[str] = Field(
        default="",
        max_length=500,
        description="Comma-separated tags"
    )
    duration: int = Field(
        ...,
        gt=0,
        le=86400,  # Max 24 hours
        description="Video duration in seconds"
    )
    publish_time: datetime = Field(
        ...,
        description="Video publish time"
    )
    category: Optional[str] = Field(
        default="Education",
        description="Video category"
    )
    
    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v: int) -> int:
        """Validate duration is reasonable."""
        if v < 1:
            raise ValueError('Duration must be at least 1 second')
        if v > 86400:  # 24 hours
            raise ValueError('Duration cannot exceed 24 hours')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: Optional[str]) -> str:
        """Validate and clean tags."""
        if v is None:
            return ""
        # Remove excessive whitespace
        tags = [tag.strip() for tag in v.split(',') if tag.strip()]
        return ','.join(tags)


class PredictionRequest(BaseModel):
    """Request model for prediction API."""
    
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default="", max_length=5000)
    tags: Optional[str] = Field(default="", max_length=500)
    duration_minutes: float = Field(..., gt=0, le=1440)  # Max 24 hours
    publish_hour: int = Field(..., ge=0, le=23)
    publish_date: str = Field(..., description="Date in YYYY-MM-DD format")
    category: Optional[str] = Field(default="Education")
    
    @field_validator('publish_date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    def to_video_metadata(self) -> VideoMetadata:
        """Convert to VideoMetadata model."""
        publish_time = datetime.strptime(self.publish_date, '%Y-%m-%d')
        publish_time = publish_time.replace(hour=self.publish_hour)
        
        return VideoMetadata(
            title=self.title,
            description=self.description or "",
            tags=self.tags or "",
            duration=int(self.duration_minutes * 60),
            publish_time=publish_time,
            category=self.category or "Education"
        )


class PredictionResponse(BaseModel):
    """Response model for prediction API."""
    
    predicted_views: int = Field(..., description="Predicted view count")
    confidence_interval: dict = Field(..., description="Confidence interval")
    suggestions: List[str] = Field(default_factory=list, description="Optimization suggestions")
    feature_importance: Optional[dict] = Field(default=None, description="Feature importance scores")


class ModelMetrics(BaseModel):
    """Model metrics validation."""
    
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    rmse: float = Field(..., ge=0, description="Root Mean Squared Error")
    r2: float = Field(..., ge=-1, le=1, description="R² Score")
    
    @field_validator('r2')
    @classmethod
    def validate_r2(cls, v: float) -> float:
        """Validate R² is in reasonable range."""
        if v < -1 or v > 1:
            raise ValueError('R² must be between -1 and 1')
        return v


def validate_video_data(data: dict) -> VideoMetadata:
    """
    Validate video data dictionary.
    
    Args:
        data: Dictionary containing video metadata
        
    Returns:
        Validated VideoMetadata object
        
    Raises:
        ValidationError: If validation fails
    """
    return VideoMetadata(**data)


def validate_prediction_request(data: dict) -> PredictionRequest:
    """
    Validate prediction request data.
    
    Args:
        data: Dictionary containing prediction request
        
    Returns:
        Validated PredictionRequest object
        
    Raises:
        ValidationError: If validation fails
    """
    return PredictionRequest(**data)
