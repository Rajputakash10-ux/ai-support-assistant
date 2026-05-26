"""
Response Engine
=================
Orchestrates the full NLP pipeline:
  1. Preprocess text
  2. Classify intent (ML model)
  3. Analyze sentiment
  4. Semantic search for best response
  5. Compose final structured response

This is the single entry point for the /chat endpoint.
"""

import json
from pathlib import Path

from backend.utils.preprocessor import extract_entities
from backend.services.intent_service import predict_intent
from backend.services.sentiment_service import analyze_sentiment

DATA_PATH = Path(__file__).resolve().parents[2] / "datasets" / "support_data.json"

# Load intent → response map once
with open(DATA_PATH) as f:
    _INTENT_RESPONSES: dict[str, str] = {
        obj["intent"]: obj["response"]
        for obj in json.load(f)["intents"]
    }

_SENTIMENT_PREFIX = {
    "negative": "We're sorry to hear you're having trouble. ",
    "positive": "Great to hear from you! ",
    "neutral": "",
}

_URGENCY_NOTE = (
    "\n\n⚠️ Your query has been flagged as urgent and escalated to a senior agent."
)


def process_query(user_message: str) -> dict:
    """
    Full pipeline execution for a single user message.
    Semantic search is skipped to keep response fast on free tier.
    """
    # --- Intent ---
    intent_result = predict_intent(user_message)
    intent = intent_result["intent"]

    # --- Sentiment ---
    sentiment_result = analyze_sentiment(user_message)

    # --- Response from intent map ---
    base_response = _INTENT_RESPONSES.get(
        intent, "I'm here to help. Could you please provide more details?"
    )

    # --- Personalize with sentiment prefix ---
    prefix = _SENTIMENT_PREFIX.get(sentiment_result["sentiment"], "")
    final_response = prefix + base_response

    if sentiment_result["is_urgent"]:
        final_response += _URGENCY_NOTE

    # --- Named Entity Recognition ---
    entities = extract_entities(user_message)

    return {
        "user_message": user_message,
        "intent": intent,
        "intent_confidence": intent_result["confidence"],
        "sentiment": sentiment_result["sentiment"],
        "sentiment_polarity": sentiment_result["polarity"],
        "is_urgent": sentiment_result["is_urgent"],
        "semantic_match_score": 0.0,
        "entities": entities,
        "response": final_response,
    }
