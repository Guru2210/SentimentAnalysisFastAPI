# Sentiment Analysis API

A lightweight, production-ready REST API built with **FastAPI** that predicts whether a given piece of text has a **Positive**, **Negative**, or **Neutral** sentiment. Predictions are powered by a fine-tuned Hugging Face Transformer model (DistilBERT).

---

##  Setup Instructions

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

If you wish to fine-tune the model from scratch on the IMDb dataset, run the training script.

> **Note:** Requires a Hugging Face Access Token set as the `HF_TOKEN` environment variable.

```bash
python train.py
```

> You do **not** need to train the model to run the API. The API automatically downloads the pre-trained weights from the Hugging Face Hub on first startup.

### 4. Start the API server

```bash
uvicorn app.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.  
Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

##  API Endpoints

### `GET /health` — Health Check

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

---

### `POST /predict` — Single Text Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love how fast the delivery was!"}'
```

---

### `POST /predict/batch` — Batch Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
           "texts": [
             "This is the best service I have ever used.",
             "The food was okay, nothing special.",
             "Terrible experience, I want my money back."
           ]
         }'
```

---

## Approach & Methodology

This project fine-tunes a **DistilBERT** model via the Hugging Face `transformers` library on the **IMDb dataset**, rather than using a traditional approach like TF-IDF with Logistic Regression.

**Why DistilBERT?**
- Transformer models natively understand deep contextual nuances and word associations, yielding significantly higher real-world accuracy.
- DistilBERT is a distilled, lighter version of BERT that keeps inference times fast enough for a production web API.

**Neutral Classification:**
Since the IMDb dataset only contains binary labels (Positive/Negative), a custom thresholding approach is applied at the API layer: if the model's confidence score falls **below 0.60**, the prediction is overridden and classified as **Neutral**.

---

##  Improvements

- **Containerization:** Package the application with Docker to ensure a perfectly reproducible environment across any machine.
- **Multi-class Training:** Fine-tune the model on a dataset that natively includes "Neutral" labels (e.g., [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)), eliminating the need for confidence score heuristics.
