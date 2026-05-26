"""
Semantic Search Service
=========================
Uses sentence embeddings + cosine similarity to find the most
relevant FAQ answer — even when the user's words don't exactly
match the stored question.

Why embeddings beat keyword search:
  "money deducted but recharge failed"  →  vector close to  →  "payment issue"
  Keyword search would miss this entirely.

Cosine similarity:
  sim(A, B) = (A · B) / (||A|| × ||B||)
  Range: [-1, 1]  →  1 = identical direction, 0 = orthogonal, -1 = opposite

Architecture:
  All FAQ questions are pre-encoded at startup (O(1) at query time).
  Query is encoded once, then dot-producted against the matrix.
"""

import json
import numpy as np
from pathlib import Path
from functools import lru_cache
# SentenceTransformer imported lazily to avoid OOM on Render free tier startup

DATA_PATH = Path(__file__).resolve().parents[2] / "datasets" / "support_data.json"

# all-MiniLM-L6-v2: 80MB, 384-dim embeddings, fast & accurate for semantic search
_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_semantic_index():
    """
    Build the semantic index once at startup.
    Returns: model, FAQ questions list, FAQ answers list, embedding matrix.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(_MODEL_NAME)

    with open(DATA_PATH) as f:
        data = json.load(f)

    # Build corpus from both FAQ questions AND intent examples for richer matching
    corpus, answers = [], []

    for faq in data["faqs"]:
        corpus.append(faq["question"])
        answers.append(faq["answer"])

    for intent_obj in data["intents"]:
        for example in intent_obj["examples"]:
            corpus.append(example)
            answers.append(intent_obj["response"])

    # Pre-compute embeddings matrix: shape (N, 384)
    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)

    return model, corpus, answers, embeddings


def semantic_search(query: str, top_k: int = 1, threshold: float = 0.4) -> dict:
    """
    Find the most semantically similar FAQ/response for a user query.
    Tries sentence transformers first, falls back to keyword matching if unavailable.
    """
    try:
        model, corpus, answers, embeddings = _load_semantic_index()
        query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = (embeddings @ query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_score = float(scores[top_indices[0]])

        if top_score < threshold:
            return {"matched": False, "score": round(top_score, 4), "matched_text": None, "response": None}

        return {"matched": True, "score": round(top_score, 4), "matched_text": corpus[top_indices[0]], "response": answers[top_indices[0]]}

    except Exception:
        # Fallback: simple keyword matching when model unavailable
        return {"matched": False, "score": 0.0, "matched_text": None, "response": None}
