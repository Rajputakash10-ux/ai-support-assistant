"""
API Schemas — Pydantic models for request/response validation.
FastAPI uses these for automatic OpenAPI docs + input validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, example="My payment failed")


class ChatResponse(BaseModel):
    user_message: str
    intent: str
    intent_confidence: float
    sentiment: str
    sentiment_polarity: float
    is_urgent: bool
    semantic_match_score: float
    entities: dict
    response: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class SentimentResponse(BaseModel):
    sentiment: str
    polarity: float
    subjectivity: float
    is_urgent: bool


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    all_scores: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
