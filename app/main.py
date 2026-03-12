from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List

# Import our new Batch schema and predict function
from app.schemas import PredictRequest, PredictResponse, BatchPredictRequest
from app.model import load_model, predict_text, predict_batch

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        print(f"Startup error: {e}")
    yield
    print("Shutting down service...")

app = FastAPI(
    title="Sentiment Analysis API",
    description="An API that predicts Positive, Negative, or Neutral sentiment.",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    try:
        result = predict_text(request.text)
        return PredictResponse(**result)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch Endpoint
@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_sentiment_batch(request: BatchPredictRequest):
    """Accepts a list of texts and returns predictions for all of them."""
    
    # Handle the edge case: Empty list
    if len(request.texts) == 0:
        raise HTTPException(
            status_code=400, 
            detail="The 'texts' list cannot be empty. Please provide at least one text string."
        )
    
    #  Process the batch
    try:
        results = predict_batch(request.texts)
        # Convert the raw dictionaries back into Pydantic models
        return [PredictResponse(**res) for res in results]
        
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))