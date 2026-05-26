"""
NLP Preprocessing Pipeline
===========================
Each step transforms raw user text into clean, normalized tokens
that ML models can understand consistently.

Why preprocessing matters:
- "Payment FAILED!!!" and "payment failed" should be treated identically
- Stopwords like "the", "is" add noise without meaning
- Lemmatization maps "running" → "run" so models see the same root form
"""

import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Always ensure NLTK data is present
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# Load spaCy model for advanced NLP (POS tagging, NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Keep negation words — "not working" ≠ "working"
_NEGATION_WORDS = {"not", "no", "never", "cannot", "can't", "won't", "don't"}
_EFFECTIVE_STOPWORDS = _stop_words - _NEGATION_WORDS


def clean_text(text: str) -> str:
    """
    Step 1 — Lowercasing + Punctuation removal
    Why: Models are case-sensitive by default. "Payment" ≠ "payment" without this.
    """
    text = text.lower().strip()
    # Remove punctuation except apostrophes (can't → can't preserved)
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """
    Step 2 — Tokenization
    Why: Splits sentence into individual units (tokens) for analysis.
    "payment failed" → ["payment", "failed"]
    Uses regex split as fallback — no punkt dependency needed.
    """
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+(?:'\w+)?\b", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Step 3 — Stopword removal
    Why: Words like "the", "is", "a" appear everywhere and carry no intent signal.
    Preserves negation words since they flip meaning entirely.
    """
    return [t for t in tokens if t not in _EFFECTIVE_STOPWORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Step 4 — Lemmatization
    Why: Reduces inflected forms to base form so the model sees one concept.
    "payments", "paying", "paid" → "payment", "pay", "pay"
    Mathematical intuition: reduces vocabulary size V, shrinks feature space.
    """
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, return_tokens: bool = False) -> str | list[str]:
    """
    Full pipeline: raw text → clean normalized string (or token list).
    Used by both the intent classifier and semantic search engine.
    """
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens if return_tokens else " ".join(tokens)


def extract_entities(text: str) -> dict:
    """
    Named Entity Recognition using spaCy.
    Extracts order IDs, amounts, dates from user messages.
    Example: "My order #12345 worth $50 placed on Monday"
    → {ORDER: ["#12345"], MONEY: ["$50"], DATE: ["Monday"]}
    """
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities
