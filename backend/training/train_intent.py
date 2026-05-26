"""
Intent Classifier Training
============================
Trains TF-IDF + ML classifiers on support intent data.
Saves the best model for production use.

Models compared:
- Logistic Regression  → fast, interpretable, strong baseline
- Naive Bayes          → probabilistic, great for text, low data
- SVM (LinearSVC)      → high-dimensional text, strong margins

TF-IDF intuition:
  TF(t,d)  = (count of t in d) / (total tokens in d)
  IDF(t)   = log(N / df(t))   — penalizes common words
  TF-IDF   = TF × IDF         — high score = important & rare word
"""

import json
import os
import sys
import joblib
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.utils.preprocessor import preprocess

DATA_PATH = Path(__file__).resolve().parents[2] / "datasets" / "support_data.json"
SAVE_DIR = Path(__file__).resolve().parents[1] / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[list[str], list[str]]:
    with open(DATA_PATH) as f:
        data = json.load(f)

    texts, labels = [], []
    for intent_obj in data["intents"]:
        for example in intent_obj["examples"]:
            texts.append(preprocess(example))
            labels.append(intent_obj["intent"])
    return texts, labels


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams capture "payment failed"
            max_features=5000,
            sublinear_tf=True,    # log(1+tf) dampens high-frequency terms
        )),
        ("clf", classifier),
    ])


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model: {name}  |  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


def train():
    print("Loading and preprocessing data...")
    X, y = load_data()
    print(f"Dataset: {len(X)} samples, {len(set(y))} intents")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": build_pipeline(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        ),
        "NaiveBayes": build_pipeline(MultinomialNB(alpha=0.1)),
        "LinearSVC": build_pipeline(LinearSVC(max_iter=2000, C=1.0, random_state=42)),
    }

    best_name, best_pipeline, best_acc = None, None, 0.0

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        evaluate(name, pipeline, X_test, y_test)

        # 5-fold cross-validation for robust estimate
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if cv_scores.mean() > best_acc:
            best_acc = cv_scores.mean()
            best_name = name
            best_pipeline = pipeline

    print(f"\n✅ Best model: {best_name} (CV Accuracy: {best_acc:.4f})")

    # Save best model + label list
    joblib.dump(best_pipeline, SAVE_DIR / "intent_classifier.pkl")
    joblib.dump(sorted(set(y)), SAVE_DIR / "intent_labels.pkl")
    print(f"Model saved to {SAVE_DIR}")

    return best_pipeline


if __name__ == "__main__":
    train()
