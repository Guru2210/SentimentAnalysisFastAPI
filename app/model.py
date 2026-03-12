from transformers import pipeline

# Global variable to hold our ML model pipeline
classifier = None

def load_model():
    """Loads the model from the Hugging Face Hub into memory."""
    global classifier
    model_id = "Guru2210/finetuning-sentiment-model-3000-samples"
    print(f"Loading model from Hugging Face Hub: {model_id}...")
    try:
        classifier = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

def predict_text(text: str) -> dict:
    """Runs a single text through the model."""
    if classifier is None:
        raise RuntimeError("Model is not loaded.")

    result = classifier(text)[0]
    raw_label = result['label']
    confidence = result['score']
    
    if confidence < 0.60:
        sentiment = "Neutral"
    elif raw_label == "LABEL_1": 
        sentiment = "Positive"
    else:                        
        sentiment = "Negative"
        
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 2)
    }

# Function to process a batch of texts ---
def predict_batch(texts: list[str]) -> list[dict]:
    """Runs a list of texts through the model efficiently."""
    if classifier is None:
        raise RuntimeError("Model is not loaded.")

    # The pipeline accepts a list natively and returns a list of dictionaries
    results = classifier(texts)
    
    output = []
    # Zip pairs our original texts with their corresponding predictions
    for text, result in zip(texts, results):
        raw_label = result['label']
        confidence = result['score']
        
        if confidence < 0.60:
            sentiment = "Neutral"
        elif raw_label == "LABEL_1":
            sentiment = "Positive"
        else:
            sentiment = "Negative"
            
        output.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 2)
        })
        
    return output