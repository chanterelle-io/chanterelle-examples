import html
import logging
import os
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIRECTORY_NAME = "imdb_distilbert_model"
MAX_LENGTH = 512


log_file = os.path.join(os.path.dirname(__file__), "handler.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    cleaned_text = html.unescape(text)
    return " ".join(cleaned_text.split())


def get_sentiment_category(score: float) -> str:
    if score > 0.8:
        return "Very Positive"
    if score > 0.6:
        return "Positive"
    if score > 0.4:
        return "Neutral"
    if score > 0.2:
        return "Negative"
    return "Very Negative"


def model_fn(model_dir: str):
    logger.info("Loading IMDB DistilBERT sentiment model from directory: %s", model_dir)
    model_path = os.path.join(model_dir, MODEL_DIRECTORY_NAME)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model directory not found at: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    logger.info("IMDB DistilBERT model loaded successfully on device: %s", device)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }


def input_fn(request_data: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Processing request data")

    review_text = request_data.get("review_text")
    if review_text is None:
        raise ValueError("Missing required field 'review_text'")
    if not isinstance(review_text, str):
        raise ValueError("'review_text' must be a string")
    if len(review_text.strip()) == 0:
        raise ValueError("'review_text' cannot be empty")

    cleaned_text = preprocess_text(review_text)
    return {
        "original_text": review_text,
        "cleaned_text": cleaned_text,
        "text_length": len(review_text),
        "word_count": len(review_text.split()),
    }


def infer_sentiment(text: str, model_bundle: Dict[str, Any]) -> Dict[str, float | str]:
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded_input = {
        key: value.to(device)
        for key, value in encoded_input.items()
    }

    with torch.no_grad():
        logits = model(**encoded_input).logits[0]

    probabilities = torch.softmax(logits, dim=0)
    positive_score = float(probabilities[1].item())
    negative_score = float(probabilities[0].item())
    predicted_index = int(torch.argmax(probabilities).item())
    predicted_label = model.config.id2label[predicted_index]
    sentiment = "Positive" if predicted_label.upper() == "POSITIVE" else "Negative"
    confidence = max(positive_score, negative_score)

    return {
        "sentiment": sentiment,
        "raw_score": positive_score,
        "confidence": confidence,
    }


def predict_fn(input_data: Dict[str, Any], model) -> Dict[str, Any]:
    logger.info("Making sentiment prediction")
    prediction = infer_sentiment(input_data["cleaned_text"], model)

    word_count = input_data["word_count"]
    text_length = input_data["text_length"]
    avg_word_length = text_length / max(word_count, 1)

    confidence_analysis = []
    sample_texts = [
        input_data["cleaned_text"][:50],
        input_data["cleaned_text"][:100],
        input_data["cleaned_text"][:200],
        input_data["cleaned_text"],
    ]
    sample_lengths = [50, 100, 200, len(input_data["cleaned_text"])]

    for sample_text, sample_length in zip(sample_texts, sample_lengths):
        normalized_sample = sample_text.strip()
        if not normalized_sample:
            continue
        sample_prediction = infer_sentiment(normalized_sample, model)
        confidence_analysis.append(
            {
                "text_length": min(sample_length, len(input_data["cleaned_text"])),
                "confidence": sample_prediction["confidence"],
                "score": sample_prediction["raw_score"],
            }
        )

    result = {
        "sentiment": prediction["sentiment"],
        "raw_score": prediction["raw_score"],
        "confidence": prediction["confidence"],
        "sentiment_category": get_sentiment_category(prediction["raw_score"]),
        "text_analysis": {
            "word_count": word_count,
            "character_count": text_length,
            "avg_word_length": avg_word_length,
        },
        "confidence_by_length": confidence_analysis,
        "original_text": (
            input_data["original_text"][:200] + "..."
            if len(input_data["original_text"]) > 200
            else input_data["original_text"]
        ),
    }

    logger.info(
        "Prediction completed: %s (confidence: %.3f)",
        result["sentiment"],
        result["confidence"],
    )
    return result


def output_fn(predictions, original_data):
    logger.info("Formatting output for sentiment prediction: %s", predictions["sentiment"])

    results_section = {
        "type": "section",
        "id": "sentiment_results",
        "title": "IMDB Movie Review Sentiment Analysis",
        "description": "Sentiment classification of movie review text using DistilBERT.",
        "items": [
            {
                "type": "table",
                "id": "prediction_results",
                "title": "Sentiment Analysis Results",
                "data": {
                    "columns": [
                        {"header": "Metric", "field": "metric"},
                        {"header": "Value", "field": "value"},
                    ],
                    "rows": [
                        {
                            "metric": "Predicted Sentiment",
                            "value": predictions["sentiment"],
                        },
                        {
                            "metric": "Confidence",
                            "value": f"{predictions['confidence']:.3f}",
                        },
                        {
                            "metric": "Positive Score",
                            "value": f"{predictions['raw_score']:.3f}",
                        },
                        {
                            "metric": "Sentiment Category",
                            "value": predictions["sentiment_category"],
                        },
                    ],
                },
            }
        ],
    }

    logger.info("Output formatting completed")
    return [results_section]