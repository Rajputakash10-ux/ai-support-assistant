#!/usr/bin/env bash
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Installing spaCy model ==="
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

echo "=== Downloading NLTK data ==="
python3 -c "
import nltk
nltk.download('punkt', quiet=False)
nltk.download('punkt_tab', quiet=False)
nltk.download('stopwords', quiet=False)
nltk.download('wordnet', quiet=False)
nltk.download('omw-1.4', quiet=False)
"

echo "=== Training intent classifier ==="
python3 -m backend.training.train_intent

echo "=== Build complete ==="
