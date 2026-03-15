import os
import torch
import numpy as np
from datasets import load_dataset
from evaluate import load
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

def main():
    # 1. Setup and Authentication
    print("Checking GPU availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Log in to Hugging Face Hub (Replaces notebook_login())
    hf_token = os.environ.get("HF_TOKEN") 
    login(token=hf_token)

    # 2. Preprocess Data
    print("Loading and preparing dataset...")
    imdb = load_dataset("stanfordnlp/imdb")

    # Create a smaller training dataset
    small_train_dataset = imdb["train"].shuffle(seed=42).select(range(3000))
    small_test_dataset = imdb["test"].shuffle(seed=42).select(range(300))

    # Set DistilBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

    # Convert samples to PyTorch tensors with padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #  Model Setup
    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2
    )
    

    # Metrics function
    def compute_metrics(eval_pred):
        load_accuracy = load("accuracy")
        load_f1 = load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    # Training the Model
    repo_name = "Guru2210/Sentiment-Analysis-DistilBERT"

    training_args = TrainingArguments(
        output_dir=repo_name,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # Push to Hub and Test Inference
    print("Pushing model to Hugging Face Hub...")
    trainer.push_to_hub()

    print("Running inference test...")
    # For local testing immediately after training, we can point the pipeline to the local output directory
    sentiment_model = pipeline("sentiment-analysis", model=repo_name, tokenizer=tokenizer)
    
    test_phrases = ["This is a fantastic movie", "This movie sucks!"]
    predictions = sentiment_model(test_phrases)
    
    for phrase, pred in zip(test_phrases, predictions):
        print(f"Text: '{phrase}' | Prediction: {pred}")

if __name__ == "__main__":
    main()