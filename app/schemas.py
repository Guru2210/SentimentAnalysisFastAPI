from pydantic import BaseModel, Field, field_validator
from typing import List

from app.model import MAX_BATCH_SIZE, MAX_TEXT_LENGTH

# Shared validator
def _non_empty_stripped(value: str) -> str:
    """Strip whitespace and reject blank strings."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("Text must be a non-empty, non-whitespace string.")
    return stripped


# Request schemas
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="The text string to analyse for sentiment.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        return _non_empty_stripped(v)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=(
            f"A list of 1–{MAX_BATCH_SIZE} text strings to analyse. "
            f"Each string must be non-empty and ≤{MAX_TEXT_LENGTH} characters."
        ),
    )

    @field_validator("texts", mode="before")
    @classmethod
    def texts_must_not_contain_blank(cls, values: list) -> list:
        cleaned = []
        for idx, item in enumerate(values):
            if not isinstance(item, str):
                raise ValueError(f"Item at index {idx} must be a string.")
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"Item at index {idx} must not be blank.")
            if len(stripped) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Item at index {idx} exceeds maximum length of "
                    f"{MAX_TEXT_LENGTH} characters."
                )
            cleaned.append(stripped)
        return cleaned


# Response schemas
class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    count: int = Field(..., description="Number of predictions returned.")
    results: List[PredictResponse]