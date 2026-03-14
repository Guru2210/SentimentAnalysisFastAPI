# Sentiment Analysis API

A lightweight, production-ready REST API built with **FastAPI** that classifies text as **Positive**, **Negative**, or **Neutral**. Inference is powered by a custom **DistilBERT** model fine-tuned on the IMDb dataset and hosted on the Hugging Face Hub.

**Model:** [`Guru2210/finetuning-sentiment-model-3000-samples`](https://huggingface.co/Guru2210/finetuning-sentiment-model-3000-samples)

---

## Project Structure

```
SentimentClassifier/
├── app/
│   ├── main.py        # FastAPI app, routes, and lifespan handler
│   ├── model.py       # Model loading, inference logic, and constants
│   └── schemas.py     # Pydantic request/response schemas
├── Train.py           # Fine-tuning script (DistilBERT on IMDb)
├── requirements.txt
└── .env               # HF_TOKEN (not committed)
```

---

## Setup Instructions

**Prerequisites:**
- Python 3.8 or higher
- A virtual environment is highly recommended

### 1. Clone the repository

```bash
git clone https://github.com/Guru2210/SentimentAnalysisFastAPI.git
cd SentimentAnalysisFastAPI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Train the model

If you wish to fine-tune the model from scratch on the IMDb dataset, set your Hugging Face token and run the training script:

```bash
# Windows
set HF_TOKEN=your_token_here

# macOS / Linux
export HF_TOKEN=your_token_here

python Train.py
```

> You do **not** need to re-train to run the API. The model weights are downloaded automatically from the Hugging Face Hub on first startup.

### 4. Start the API server

```bash
uvicorn app.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.  
Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

### `GET /health` — Health Check

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /v1/predict` — Single Text Prediction

```bash
curl -X POST "http://127.0.0.1:8000/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love how fast the delivery was!"}'
```

**Response:**
```json
{
  "text": "I absolutely love how fast the delivery was!",
  "sentiment": "Positive",
  "confidence": 0.9983
}
```

> **Constraints:** Text must be non-empty and ≤ 512 characters.

---

### `POST /v1/predict/batch` — Batch Prediction

```bash
curl -X POST "http://127.0.0.1:8000/v1/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
           "texts": [
             "This is the best service I have ever used.",
             "The food was okay, nothing special.",
             "Terrible experience, I want my money back."
           ]
         }'
```

**Response:**
```json
{
  "count": 3,
  "results": [
    { "text": "This is the best service I have ever used.", "sentiment": "Positive", "confidence": 0.9971 },
    { "text": "The food was okay, nothing special.", "sentiment": "Neutral", "confidence": 0.5312 },
    { "text": "Terrible experience, I want my money back.", "sentiment": "Negative", "confidence": 0.9945 }
  ]
}
```

> **Constraints:** 1–256 texts per request; each text must be non-empty and ≤ 512 characters.

---

## Approach & Methodology

This project fine-tunes a **DistilBERT** model via the Hugging Face `transformers` library on the **IMDb dataset**, rather than a traditional approach like TF-IDF + Logistic Regression.

**Why DistilBERT?**
- Transformer models understand deep contextual nuances and word associations, yielding significantly higher real-world accuracy.
- DistilBERT is a distilled, lighter version of BERT that keeps inference fast enough for a production web API.

**Training details:**
- Dataset: 3,000 training samples / 300 test samples (shuffled from Stanford IMDb)
- Epochs: 2 | Learning rate: 2e-5 | Batch size: 16 | Weight decay: 0.01
- Metrics tracked: Accuracy & F1

**Neutral classification:**  
The IMDb dataset only has binary labels (Positive/Negative). A confidence-based heuristic is applied at inference time: if the model's confidence score falls **below 0.60**, the prediction is overridden to **Neutral**.

---

## Potential Improvements

- **Containerization:** Package the application with Docker for a reproducible deployment environment.
- **Multi-class Training:** Fine-tune on a dataset with native "Neutral" labels (e.g., [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)) to replace the confidence heuristic.
- **Async inference:** Move model inference to a background thread pool to avoid blocking the event loop under high concurrency.
- **Rate limiting:** Add per-client rate limiting middleware to protect against abuse.
