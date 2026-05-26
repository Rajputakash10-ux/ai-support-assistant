"""
Sentiment Analysis Service
============================
Analyzes emotional tone of user messages.

Business value:
- Route angry users (negative) to senior agents immediately
- Track satisfaction trends over time
- Trigger escalation workflows automatically

TextBlob polarity: [-1.0 (very negative) → +1.0 (very positive)]
Subjectivity:      [0.0 (factual) → 1.0 (opinionated)]
"""

from textblob import TextBlob


_THRESHOLDS = {"positive": 0.1, "negative": -0.1}

_URGENCY_KEYWORDS = {
    "urgent", "immediately", "asap", "emergency", "critical",
    "frustrated", "angry", "terrible", "worst", "scam", "fraud"
}


def analyze_sentiment(text: str) -> dict:
    """
    Returns sentiment label, polarity score, subjectivity, and urgency flag.

    Urgency detection: negative sentiment + strong keywords → escalate immediately.
    """
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity >= _THRESHOLDS["positive"]:
        label = "positive"
    elif polarity <= _THRESHOLDS["negative"]:
        label = "negative"
    else:
        label = "neutral"

    words = set(text.lower().split())
    is_urgent = label == "negative" and bool(words & _URGENCY_KEYWORDS)

    return {
        "sentiment": label,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "is_urgent": is_urgent,
    }
