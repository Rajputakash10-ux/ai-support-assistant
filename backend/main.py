"""
FastAPI Application Entry Point
==================================
Production-style setup with:
- Lifespan context for model warm-up (avoids cold-start latency on first request)
- CORS middleware for React frontend
- Global exception handler
- OpenAPI docs at /docs
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from backend.api.routes import router
from backend.services.intent_service import _load_model
from backend.services.semantic_service import _load_semantic_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up ML models at startup. Errors are caught so the server
    still starts even if a model fails to load.
    """
    print("🚀 Warming up NLP models...")
    try:
        _load_model()
        print("✅ Intent classifier loaded.")
    except Exception as e:
        print(f"⚠️ Intent classifier failed to load: {e}")
    try:
        _load_semantic_index()
        print("✅ Semantic index loaded.")
    except Exception as e:
        print(f"⚠️ Semantic index failed to load: {e}")
    print("✅ Server ready.")
    yield
    print("🛑 Shutting down.")


app = FastAPI(
    title=os.getenv("APP_NAME", "AI Support Assistant"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="AI-powered intelligent customer support backend with intent detection, sentiment analysis, and semantic search.",
    lifespan=lifespan,
)

# CORS — allow React dev server and production frontend
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred.", "error": str(exc)},
    )


@app.get("/", tags=["Root"])
async def root():
    return {"message": "AI Support Assistant API", "docs": "/docs"}
