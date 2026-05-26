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
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from backend.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting...")
    yield
    print("Shutting down.")


app = FastAPI(
    title=os.getenv("APP_NAME", "AI Support Assistant"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="AI-powered intelligent customer support backend v2.",
    lifespan=lifespan,
)

# CORS — allow all origins for portfolio/demo deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
