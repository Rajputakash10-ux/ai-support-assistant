"""
NLP Preprocessing Pipeline
===========================
Each step transforms raw user text into clean, normalized tokens
that ML models can understand consistently.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
_NEGATION_WORDS = {"not", "no", "never", "cannot", "can't", "won't", "don't"}
_EFFECTIVE_STOPWORDS = _stop_words - _NEGATION_WORDS


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+(?:'\w+)?\b", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _EFFECTIVE_STOPWORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, return_tokens: bool = False) -> str | list[str]:
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens if return_tokens else " ".join(tokens)


def extract_entities(text: str) -> dict:
    """Lazy-load spaCy to avoid slow startup on free tier."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {}
        doc = nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return entities
    except Exception:
        return {}
