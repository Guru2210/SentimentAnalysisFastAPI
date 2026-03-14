import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
from app.model import load_model, predict_text, predict_batch

# Logging
logger = logging.getLogger(__name__)

# Lifespan: fatal if model fails to load


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; any failure aborts the process immediately."""
    logger.info("Starting up — loading sentiment model…")
    load_model()                   # Raises RuntimeError on failure → process exits
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down service.")

# App
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts Positive, Negative, or Neutral sentiment for text input.",
    version="1.0.0",
    lifespan=lifespan,
)

# Exception handlers

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Map validation errors raised inside route handlers to HTTP 422."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    """Map model-unavailable errors to HTTP 503."""
    logger.error("RuntimeError: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Model is currently unavailable. Try again later."},
    )


# Routes

@app.get("/health", tags=["ops"])
def health_check():
    """Liveness probe — confirms the service is reachable."""
    return {"status": "ok"}


@app.post(
    "/v1/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Single-text sentiment prediction",
)
def predict_sentiment(request: PredictRequest):
    """
    Analyse the sentiment of a single text string.

    Returns one of **Positive**, **Negative**, or **Neutral**,
    along with a confidence score in [0, 1].
    """
    try:
        result = predict_text(request.text)
    except (ValueError, RuntimeError):
        raise                        # Delegated to exception handlers above
    except Exception as exc:
        logger.exception("Unexpected error in /v1/predict")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    return PredictResponse(**result)


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
    summary="Batch sentiment prediction",
)
def predict_sentiment_batch(request: BatchPredictRequest):
    """
    Analyse the sentiment of a list of text strings in one call.

    - Accepts 1 to `MAX_BATCH_SIZE` strings.
    - Returns predictions in the same order as the input list.
    - The response includes a `count` field for easy pagination/logging.
    """
    try:
        raw_results = predict_batch(request.texts)
    except (ValueError, RuntimeError):
        raise                        # Delegated to exception handlers above
    except Exception as exc:
        logger.exception("Unexpected error in /v1/predict/batch")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    predictions = [PredictResponse(**r) for r in raw_results]
    return BatchPredictResponse(count=len(predictions), results=predictions)