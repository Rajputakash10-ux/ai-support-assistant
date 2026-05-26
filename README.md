# 🤖 AI-Powered Intelligent Support Assistant

> A production-grade NLP system that understands user queries, detects intent, analyzes sentiment, performs semantic search, and returns intelligent context-aware responses.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)

---

## 🎯 What It Does

| Input | Output |
|-------|--------|
| `"My payment failed but money got deducted"` | **Intent:** Payment Issue · **Sentiment:** Negative · **Response:** Context-aware resolution message |

---

## 🏗️ Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────┐
│           FastAPI Backend               │
│  ┌──────────────────────────────────┐   │
│  │       NLP Preprocessing          │   │
│  │  lowercase → tokenize → lemma    │   │
│  └──────────────┬───────────────────┘   │
│                 │                       │
│     ┌───────────┼───────────┐           │
│     ▼           ▼           ▼           │
│  Intent      Sentiment   Semantic       │
│  Classifier  Analysis    Search         │
│  (TF-IDF     (TextBlob)  (Sentence      │
│  + SVM)                  Transformers)  │
│     └───────────┼───────────┘           │
│                 ▼                       │
│         Response Engine                 │
│    (orchestrates all results)           │
└─────────────────────────────────────────┘
     │
     ▼
React Frontend (Chat UI)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Clone & Setup Backend

```bash
git clone https://github.com/yourusername/ai-support-assistant.git
cd ai-support-assistant

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train the Intent Classifier

```bash
python -m backend.training.train_intent
```

### 3. Start the API Server

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: http://localhost:5173

---

## 📡 API Reference

### `POST /api/v1/chat`
Full NLP pipeline — main endpoint.

**Request:**
```json
{ "message": "My payment failed but money got deducted" }
```

**Response:**
```json
{
  "intent": "payment_issue",
  "intent_confidence": 0.92,
  "sentiment": "negative",
  "sentiment_polarity": -0.35,
  "is_urgent": false,
  "semantic_match_score": 0.87,
  "entities": {},
  "response": "We're sorry to hear you're having trouble. We detected a payment issue..."
}
```

### `POST /api/v1/predict`
Intent classification only.

### `POST /api/v1/sentiment`
Sentiment analysis only.

### `GET /api/v1/health`
Liveness probe for deployment platforms.

---

## 🧠 NLP Concepts Explained

### TF-IDF (Term Frequency–Inverse Document Frequency)
```
TF(t,d)  = count(t in d) / total_tokens(d)
IDF(t)   = log(N / df(t))
TF-IDF   = TF × IDF
```
High score = word is frequent in this document but rare across all documents → meaningful signal.

### Cosine Similarity (Semantic Search)
```
sim(A, B) = (A · B) / (||A|| × ||B||)
```
Measures angle between embedding vectors. Score of 1 = identical meaning, 0 = unrelated.

### Why Sentence Embeddings Beat Keywords
- `"money deducted but recharge failed"` → embedding close to `"payment issue"`
- Keyword search would return 0 matches. Semantic search returns 0.87 similarity.

---

## 🗂️ Project Structure

```
ai-support-assistant/
├── backend/
│   ├── api/
│   │   ├── routes.py          # FastAPI endpoints
│   │   └── schemas.py         # Pydantic request/response models
│   ├── services/
│   │   ├── intent_service.py  # ML intent classifier
│   │   ├── sentiment_service.py
│   │   ├── semantic_service.py # Sentence embeddings + cosine sim
│   │   └── response_engine.py  # Orchestration layer
│   ├── utils/
│   │   └── preprocessor.py    # NLP pipeline (clean→tokenize→lemma)
│   ├── training/
│   │   └── train_intent.py    # TF-IDF + ML training script
│   └── main.py                # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── components/        # ChatMessage, MetaBadges, TypingIndicator
│       ├── utils/api.js       # Backend API client
│       └── App.jsx            # Main chat interface
├── datasets/
│   └── support_data.json      # Intent examples + FAQ data
├── notebooks/
│   └── exploration.ipynb      # EDA + model training walkthrough
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| NLP Preprocessing | NLTK, spaCy | Industry standard, production-proven |
| Intent Classification | scikit-learn, TF-IDF + SVM | Fast, interpretable, high accuracy on text |
| Sentiment Analysis | TextBlob | Lightweight, no GPU needed |
| Semantic Search | SentenceTransformers | State-of-the-art embeddings, 80MB model |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Frontend | React + Vite | Fast, component-based, recruiter-friendly |

---

## 🔮 Upgrade Path (Advanced NLP)

```
Current Stack          →    Advanced Upgrade
─────────────────────────────────────────────
TF-IDF + SVM           →    Fine-tuned BERT / DistilBERT
TextBlob sentiment     →    HuggingFace transformers pipeline
Cosine similarity      →    FAISS vector index (million-scale)
JSON FAQ store         →    Pinecone / Weaviate vector DB
Rule-based responses   →    RAG (Retrieval-Augmented Generation)
Single model           →    LangChain AI Agents
```

---

## ☁️ Deployment

### Backend → Render / Railway
```bash
# Procfile
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```
Set env vars: `ALLOWED_ORIGINS=https://your-frontend.vercel.app`

### Frontend → Vercel
```bash
cd frontend
vercel --prod
```
Set env: `VITE_API_URL=https://your-backend.render.com`

---

## 🎤 Interview Talking Points

**"Walk me through your NLP pipeline."**
> "User text goes through 5 preprocessing steps — lowercase, punctuation removal, tokenization, stopword removal, and lemmatization. This normalized text feeds into a TF-IDF vectorizer that converts it to a sparse feature vector, which a LinearSVC classifies into one of 6 intents. In parallel, a SentenceTransformer encodes the raw text into a 384-dimensional dense vector for semantic FAQ matching using cosine similarity."

**"Why TF-IDF over bag-of-words?"**
> "TF-IDF penalizes words that appear in every document — like 'the', 'is' — so the model focuses on discriminative terms. Bag-of-words treats all words equally, which hurts precision."

**"How would you scale this to 10M users?"**
> "Replace the in-memory semantic index with FAISS for ANN search, add Redis caching for repeated queries, deploy the FastAPI app behind a load balancer on ECS/Kubernetes, and use async workers for model inference."

---

## 📄 License

MIT — free to use for portfolio and commercial projects.
