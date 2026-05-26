"""
API Routes
============
All endpoints are async for non-blocking I/O under load.
Each endpoint is single-responsibility and independently testable.
"""

from fastapi import APIRouter, HTTPException
from backend.api.schemas import (
    ChatRequest, ChatResponse,
    SentimentRequest, SentimentResponse,
    PredictRequest, PredictResponse,
    HealthResponse,
)
from backend.services.response_engine import process_query
from backend.services.intent_service import predict_intent
from backend.services.sentiment_service import analyze_sentiment

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — used by load balancers and deployment platforms."""
    from pathlib import Path
    model_exists = Path("backend/models/saved/intent_classifier.pkl").exists()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=model_exists,
    )


@router.post("/chat", response_model=ChatResponse, tags=["Core"])
async def chat(request: ChatRequest):
    """
    Main endpoint — runs the full NLP pipeline on a user message.
    Returns intent, sentiment, entities, and a context-aware response.
    """
    try:
        result = process_query(request.message)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.post("/predict", response_model=PredictResponse, tags=["NLP"])
async def predict(request: PredictRequest):
    """Intent classification only — useful for analytics dashboards."""
    try:
        result = predict_intent(request.text)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment", response_model=SentimentResponse, tags=["NLP"])
async def sentiment(request: SentimentRequest):
    """Sentiment analysis only — useful for review monitoring pipelines."""
    try:
        result = analyze_sentiment(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
