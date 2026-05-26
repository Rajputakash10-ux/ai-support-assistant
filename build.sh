#!/usr/bin/env bash
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Installing spaCy model ==="
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

echo "=== Downloading NLTK data ==="
python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
"

echo "=== Build complete ==="
