import json
import logging
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import re
from typing import Dict, List, Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging to write to a file
log_file = os.path.join(os.path.dirname(__file__), 'handler.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This writes to stderr, not stdout
    ]
)
logger = logging.getLogger(__name__)

# Custom TensorFlow logging - redirect TF warnings to our logger
class TensorFlowLogHandler(logging.Handler):
    def emit(self, record):
        # Forward TensorFlow logs to our main logger
        if record.levelno >= logging.WARNING:
            logger.warning(f"TensorFlow: {record.getMessage()}")
        elif record.levelno >= logging.INFO:
            logger.info(f"TensorFlow: {record.getMessage()}")

# Redirect TensorFlow's Python logger to our custom handler
tf_logger = tf.get_logger()
tf_logger.handlers.clear()  # Remove default handlers
tf_handler = TensorFlowLogHandler()
tf_handler.setLevel(logging.WARNING)  # Only capture warnings and errors
tf_logger.addHandler(tf_handler)
tf_logger.setLevel(logging.WARNING)

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text data for sentiment analysis
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (keep basic punctuation)
    text = re.sub(r'[^a-zA-Z\s\.\!\?\,]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def get_sentiment_category(score: float) -> str:
    """Convert raw score to sentiment category with confidence levels"""
    if score > 0.8:
        return "Very Positive"
    elif score > 0.6:
        return "Positive"
    elif score > 0.4:
        return "Neutral"
    elif score > 0.2:
        return "Negative"
    else:
        return "Very Negative"

def model_fn(model_dir: str):
    """Load the IMDB sentiment classification model from the specified directory"""
    logger.info(f"Loading IMDB sentiment classification model from directory: {model_dir}")
    try:
        model_path = os.path.join(model_dir, 'imdb_tfhub_model')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at: {model_path}")
        
        # model = keras.models.load_model(model_path)
        model = keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
        logger.info("IMDB sentiment model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")

def input_fn(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and preprocess review text from request data"""
    logger.info(f"Processing request_data: {request_data}")
    try:
        # Extract review text from the request
        review_text = request_data.get('review_text')
        
        if review_text is None:
            raise ValueError("Missing required field 'review_text'")
        
        if not isinstance(review_text, str):
            raise ValueError("'review_text' must be a string")
        
        if len(review_text.strip()) == 0:
            raise ValueError("'review_text' cannot be empty")
        
        # Preprocess the text
        cleaned_text = preprocess_text(review_text)
        
        # Prepare input data
        input_data = {
            'original_text': review_text,
            'cleaned_text': cleaned_text,
            'text_length': len(review_text),
            'word_count': len(review_text.split())
        }
        
        logger.info(f"Processed review with {input_data['word_count']} words")
        return input_data
        
    except Exception as e:
        logger.error(f"Error processing input_data: {e}")
        raise ValueError(f"Could not process input data: {e}")

def predict_fn(input_data: Dict[str, Any], model) -> Dict[str, Any]:
    """Make sentiment prediction using the IMDB classification model"""
    logger.info(f"Making sentiment prediction for review")
    
    try:
        # Get the cleaned text for prediction
        text_for_prediction = input_data['cleaned_text']
        
        # Make prediction using the model
        # The TFHub model expects raw text input
        raw_score = model.predict([text_for_prediction])[0][0]
        
        # Convert to sentiment
        sentiment = "Positive" if raw_score > 0.5 else "Negative"
        confidence = float(raw_score if raw_score > 0.5 else 1 - raw_score)
        sentiment_category = get_sentiment_category(raw_score)
        
        # Generate some analysis data for visualization
        # Analyze text characteristics
        word_count = input_data['word_count']
        text_length = input_data['text_length']
        avg_word_length = text_length / max(word_count, 1)
        
        # Simple keyword analysis (positive/negative indicators)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 
                         'perfect', 'love', 'best', 'brilliant', 'outstanding', 'superb']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 
                         'boring', 'disappointing', 'poor', 'weak', 'pathetic', 'annoying']
        
        text_lower = input_data['cleaned_text'].lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Create confidence analysis across different text lengths
        confidence_analysis = []
        sample_texts = [
            input_data['cleaned_text'][:50] + "..." if len(input_data['cleaned_text']) > 50 else input_data['cleaned_text'],
            input_data['cleaned_text'][:100] + "..." if len(input_data['cleaned_text']) > 100 else input_data['cleaned_text'],
            input_data['cleaned_text'][:200] + "..." if len(input_data['cleaned_text']) > 200 else input_data['cleaned_text'],
            input_data['cleaned_text']
        ]
        
        text_lengths = [50, 100, 200, len(input_data['cleaned_text'])]
        
        for i, sample_text in enumerate(sample_texts):
            if len(sample_text.strip()) > 0:
                sample_score = model.predict([sample_text])[0][0]
                sample_confidence = float(sample_score if sample_score > 0.5 else 1 - sample_score)
                confidence_analysis.append({
                    'text_length': min(text_lengths[i], len(input_data['cleaned_text'])),
                    'confidence': sample_confidence,
                    'score': float(sample_score)
                })
        
        result = {
            'sentiment': sentiment,
            'raw_score': float(raw_score),
            'confidence': confidence,
            'sentiment_category': sentiment_category,
            'text_analysis': {
                'word_count': word_count,
                'character_count': text_length,
                'avg_word_length': avg_word_length,
                'positive_keywords': positive_count,
                'negative_keywords': negative_count
            },
            'confidence_by_length': confidence_analysis,
            'original_text': input_data['original_text'][:200] + "..." if len(input_data['original_text']) > 200 else input_data['original_text']
        }
        
        logger.info(f"Prediction completed: {sentiment} (confidence: {confidence:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise RuntimeError(f"Could not make prediction: {e}")

def output_fn(predictions: Dict[str, Any], original_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format output for IMDB sentiment classification results"""
    logger.info(f"Formatting output for sentiment prediction: {predictions['sentiment']}")
    
    # Main results section
    results_section = {
        "type": "section",
        "id": "sentiment_results",
        "title": "IMDB Movie Review Sentiment Analysis",
        # "color": "green" if predictions['sentiment'] == "Positive" else "red",
        "description": "AI-powered sentiment classification of movie review text.",
        "items": [
            # {
            #     "type": "table",
            #     "id": "prediction_results",
            #     "title": "Sentiment Analysis Results",
            #     "data": {
            #         "columns": [
            #             {"header": "Metric", "field": "metric"},
            #             {"header": "Value", "field": "value"}
            #         ],
            #         "rows": [
            #             {
            #                 "metric": "Predicted Sentiment",
            #                 "value": predictions['sentiment']
            #             },
            #             # {
            #             #     "metric": "Confidence Level",
            #             #     "value": f"{predictions['confidence']:.1%}"
            #             # },
            #             # {
            #             #     "metric": "Sentiment Category",
            #             #     "value": predictions['sentiment_category']
            #             # },
            #             # {
            #             #     "metric": "Raw Model Score",
            #             #     "value": f"{predictions['raw_score']:.3f}"
            #             # }
            #         ]
            #     }
            # },
            # {
            #     "type": "text",
            #     "id": "review_preview",
            #     "title": "Review Text (Preview)",
            #     "content": predictions['original_text']
            # }
        ]
    }
    
    # # Text analysis section
    # text_analysis = predictions['text_analysis']
    # analysis_section = {
    #     "type": "section",
    #     "id": "text_analysis",
    #     "title": "Text Analysis",
    #     "description": "Statistical analysis of the review text characteristics.",
    #     "items": [
    #         {
    #             "type": "table",
    #             "id": "text_stats",
    #             "title": "Text Statistics",
    #             "data": {
    #                 "columns": [
    #                     {"header": "Statistic", "field": "statistic"},
    #                     {"header": "Value", "field": "value"}
    #                 ],
    #                 "rows": [
    #                     {
    #                         "statistic": "Word Count",
    #                         "value": str(text_analysis['word_count'])
    #                     },
    #                     {
    #                         "statistic": "Character Count",
    #                         "value": str(text_analysis['character_count'])
    #                     },
    #                     {
    #                         "statistic": "Average Word Length",
    #                         "value": f"{text_analysis['avg_word_length']:.1f} characters"
    #                     },
    #                     {
    #                         "statistic": "Positive Keywords Found",
    #                         "value": str(text_analysis['positive_keywords'])
    #                     },
    #                     {
    #                         "statistic": "Negative Keywords Found",
    #                         "value": str(text_analysis['negative_keywords'])
    #                     }
    #                 ]
    #             }
    #         }
    #     ]
    # }
    
    # # Confidence analysis section
    # confidence_data = predictions.get('confidence_by_length', [])
    # if confidence_data:
    #     confidence_section = {
    #         "type": "section",
    #         "id": "confidence_analysis",
    #         "title": "Confidence Analysis",
    #         "description": "How prediction confidence changes with text length.",
    #         "items": [
    #             {
    #                 "type": "line_chart",
    #                 "id": "confidence_by_length",
    #                 "title": "Confidence vs Text Length",
    #                 "description": "Model confidence at different text lengths.",
    #                 "data": {
    #                     "lines": [
    #                         {
    #                             "id": "confidence_curve",
    #                             "points": [
    #                                 {"x": point['text_length'], "y": round(point['confidence'], 3)}
    #                                 for point in confidence_data
    #                             ],
    #                             "style": {
    #                                 "color": "blue",
    #                                 "width": 3
    #                             }
    #                         }
    #                     ],
    #                     "axis": {
    #                         "x": {"label": "Text Length (characters)"},
    #                         "y": {"label": "Prediction Confidence"}
    #                     }
    #                 }
    #             }
    #         ]
    #     }
    # else:
    #     confidence_section = None
    
    # # Keyword analysis chart
    # if text_analysis['positive_keywords'] > 0 or text_analysis['negative_keywords'] > 0:
    #     keyword_section = {
    #         "type": "section",
    #         "id": "keyword_analysis",
    #         "title": "Keyword Analysis",
    #         "description": "Distribution of positive and negative sentiment keywords.",
    #         "items": [
    #             {
    #                 "type": "bar_chart",
    #                 "id": "keyword_distribution",
    #                 "title": "Sentiment Keywords Found",
    #                 "description": "Count of positive vs negative keywords in the text.",
    #                 "data": {
    #                     "bars": [
    #                         {
    #                             "label": "Positive Keywords",
    #                             "value": text_analysis['positive_keywords'],
    #                             "color": "green"
    #                         },
    #                         {
    #                             "label": "Negative Keywords", 
    #                             "value": text_analysis['negative_keywords'],
    #                             "color": "red"
    #                         }
    #                     ],
    #                     "axis": {
    #                         "x": {"label": "Keyword Type"},
    #                         "y": {"label": "Count"}
    #                     },
    #                     "orientation": "vertical"
    #                 }
    #             }
    #         ]
    #     }
    # else:
    #     keyword_section = None
    
    # Compile response sections
    response = [results_section] #, analysis_section]
    
    # if confidence_section:
    #     response.append(confidence_section)
    
    # if keyword_section:
    #     response.append(keyword_section)
    
    logger.info(f"Output formatting completed with {len(response)} sections")
    logger.info(f"Output formatting completed with: {response}")

    return response
