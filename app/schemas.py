from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    text: str = Field(..., description="The text string to analyze for sentiment")

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

# Schema for the batch request 
class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of text strings to analyze")