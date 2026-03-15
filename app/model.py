import logging
import threading
from transformers import pipeline

# Public API surface

__all__ = ["load_model", "predict_text", "predict_batch"]

# Configuration

MODEL_ID = "Guru2210/Sentiment-Analysis-DistilBERT"
NEUTRAL_CONFIDENCE_THRESHOLD = 0.60
POSITIVE_LABEL = "LABEL_1"
MAX_BATCH_SIZE = 256   # Hard cap to prevent OOM on large payloads

# Character-based proxy for token length. Safe for English/Latin text.
# NOTE: For multilingual/CJK input, one character can map to multiple tokens.
# If you serve non-Latin languages, replace this with a tokenizer.encode()
# length check against the model's true max_position_embeddings (typically 512).
MAX_TEXT_LENGTH = 512

# Logging
logger = logging.getLogger(__name__)

# Thread-safe singleton loader
_classifier = None
_lock = threading.Lock()


def load_model() -> None:
    """
    Load the sentiment model into memory (idempotent, thread-safe).
    Raises RuntimeError if the model cannot be loaded.
    """
    global _classifier

    # Fast path: already loaded — no lock needed for a read
    if _classifier is not None:
        logger.info("Model already loaded; skipping reload.")
        return

    with _lock:
        # Double-checked locking: re-test inside the lock
        if _classifier is not None:
            return

        logger.info("Loading model '%s' from Hugging Face Hub…", MODEL_ID)
        try:
            _classifier = pipeline(
                "sentiment-analysis",
                model=MODEL_ID,
                tokenizer=MODEL_ID,
                return_token_type_ids=False
            )
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            raise RuntimeError(f"Could not load model '{MODEL_ID}'.") from exc


def _get_classifier():
    """Return the loaded classifier or raise a clear error."""
    if _classifier is None:
        raise RuntimeError(
            "Model is not loaded. Call load_model() before running predictions."
        )
    return _classifier


# Input validation

def _validate_text(text: str) -> None:
    """
    Raise ValueError for invalid inputs.

    Kept here intentionally: model.py is a standalone library usable outside
    FastAPI. Schema-level validation in schemas.py is a first line of defence,
    but this layer protects direct Python callers too.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Input text exceeds maximum length of {MAX_TEXT_LENGTH} characters "
            f"(got {len(text)})."
        )


# Internal label mapper (single source of truth)

def _map_result(text: str, result: dict) -> dict:
    """
    Convert a raw pipeline result into a structured prediction dict.

    Args:
        text:   The original input string.
        result: A dict with 'label' and 'score' from the pipeline.

    Returns:
        dict with keys: text, sentiment, confidence.
    """
    raw_label: str = result["label"]
    confidence: float = result["score"]

    if confidence < NEUTRAL_CONFIDENCE_THRESHOLD:
        sentiment = "Neutral"
    elif raw_label == POSITIVE_LABEL:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
    }

# Public API

def predict_text(text: str) -> dict:
    """
    Run sentiment analysis on a single string.

    Delegates to predict_batch to guarantee identical behaviour and avoid
    duplicating classifier invocation logic.

    Args:
        text: The input text to classify.

    Returns:
        dict with keys: text, sentiment, confidence.

    Raises:
        ValueError:   If the input is empty or too long.
        RuntimeError: If the model has not been loaded.
    """
    return predict_batch([text])[0]


def predict_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Run sentiment analysis on a list of strings efficiently.

    Args:
        texts:      List of input strings to classify.
        batch_size: Number of samples passed to the pipeline at once.
                    Tune based on available memory (default: 32).

    Returns:
        List of dicts, each with keys: text, sentiment, confidence.
        Order matches the input list.

    Raises:
        ValueError:   If any input is empty/too long, or the list exceeds
                      MAX_BATCH_SIZE.
        RuntimeError: If the model has not been loaded.
    """
    if not texts:
        return []

    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(texts)} exceeds the maximum allowed "
            f"({MAX_BATCH_SIZE}). Split into smaller chunks."
        )

    for idx, text in enumerate(texts):
        try:
            _validate_text(text)
        except ValueError as exc:
            raise ValueError(f"Invalid input at index {idx}: {exc}") from exc

    clf = _get_classifier()
    raw_results = clf(texts, batch_size=batch_size)

    logger.debug("predict_batch | n=%d | batch_size=%d", len(texts), batch_size)

    return [_map_result(text, result) for text, result in zip(texts, raw_results)]