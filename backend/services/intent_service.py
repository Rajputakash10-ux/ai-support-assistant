"""
Intent Classification Service
================================
Loads the trained TF-IDF + ML pipeline and exposes a predict() method.
Falls back to semantic matching if confidence is below threshold.
"""

import joblib
import numpy as np
from pathlib import Path
from functools import lru_cache

from backend.utils.preprocessor import preprocess

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "saved"


@lru_cache(maxsize=1)
def _load_model():
    """Lazy-load model once and cache in memory (singleton pattern)."""
    clf = joblib.load(MODEL_DIR / "intent_classifier.pkl")
    labels = joblib.load(MODEL_DIR / "intent_labels.pkl")
    return clf, labels


def predict_intent(text: str, threshold: float = 0.35) -> dict:
    """
    Predict intent from raw user text.

    Returns:
        intent    — predicted class label
        confidence — probability score [0, 1]
        all_scores — scores for every intent (useful for debugging)
    """
    clf, labels = _load_model()
    cleaned = preprocess(text)

    # LinearSVC uses decision_function; others use predict_proba
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba([cleaned])[0]
        confidence = float(np.max(proba))
        intent = clf.classes_[np.argmax(proba)]
        all_scores = dict(zip(clf.classes_, proba.tolist()))
    else:
        scores = clf.decision_function([cleaned])[0]
        # Softmax to convert margins → probabilities
        exp_scores = np.exp(scores - np.max(scores))
        proba = exp_scores / exp_scores.sum()
        confidence = float(np.max(proba))
        intent = clf.classes_[np.argmax(proba)]
        all_scores = dict(zip(clf.classes_, proba.tolist()))

    if confidence < threshold:
        intent = "general_inquiry"

    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
    }
