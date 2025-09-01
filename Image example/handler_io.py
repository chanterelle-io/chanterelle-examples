import json
import logging
import os
import sys
import joblib
import numpy as np
# try:
from PIL import Image
# except ImportError:
#     Image = None

# Suppress TensorFlow warnings from console but allow them to be logged
# TensorFlow warnings will be captured by our custom logger below

import tensorflow as tf

# Set up logging to write to a file FIRST
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

# Define digit classes and colors for visualization
DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DIGIT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def model_fn(model_dir):
    """Load the MNIST classification model from the specified directory"""
    logger.info(f"Loading MNIST classification model from directory: {model_dir}")
    
    try:
        # Load TensorFlow/Keras model directly
        model_path = os.path.join(model_dir, 'mnist_classification_model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Keras model loaded successfully")
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")

def input_fn(request_data):
    """Extract image data from request - primarily expects file paths to image files"""
    logger.info(f"Processing request_data with keys: {list(request_data.keys()) if isinstance(request_data, dict) else 'not a dict'}")
    logger.info(f"Processing request_data: {request_data}")
    # try:
    image_data = None
    
    # Primary format: file path to an image
    if 'number_image' in request_data:
        # request_data for files are dict: {name, path}
        image_path = request_data.get('number_image').get('path')
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        if Image is None:
            raise ValueError("PIL not available for image processing")
        
        try:
            # Load image using PIL
            logger.info(f"Loading image from file: {image_path}")
            pil_image = Image.open(image_path)
            original_mode = pil_image.mode
            
            # Convert to grayscale and resize to 28x28
            pil_image_processed = pil_image.convert('L').resize((28, 28))
            
            # Save the processed image for reference
            base_name = request_data.get('number_image').get('name')
            processed_image_path = os.path.join(os.path.dirname(__file__), f"processed_images/{base_name}_processed_28x28.png")
            pil_image_processed.save(processed_image_path)
            logger.info(f"Saved processed image to: {processed_image_path}")
            
            # Convert to numpy array
            image_data = np.array(pil_image_processed)
            
            logger.info(f"Loaded image from {image_path}, original mode: {original_mode}, processed and saved to: {processed_image_path}")
            
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            raise ValueError(f"Could not load image file: {e}")
    else:
        raise ValueError("No valid image data found. Expected 'image_path'")
    
    if image_data is None:
        raise ValueError("Could not extract image data from request")
    
    # Ensure image is in the right format
    image_data = np.array(image_data, dtype=np.float32)
    
    # Normalize if needed (values should be 0-255 or 0-1)
    if np.max(image_data) > 1:
        image_data = image_data / 255.0
    
    input_data = {
        'image': image_data
    }
    
    logger.info(f"Processed image with shape: {image_data.shape}, min: {np.min(image_data):.3f}, max: {np.max(image_data):.3f}")
    return input_data
        
    # except Exception as e:
    #     logger.error(f"Error processing input_data: {e}")
    #     raise ValueError(f"Could not process input data: {e}")

def predict_fn(input_data, model):
    """Make prediction using the MNIST classification model"""
    logger.info(f"Making prediction with image shape: {input_data['image'].shape}")
    
    try:
        # Get image from input data
        image = input_data['image']
        
        # Prepare image for model (ensure correct shape and normalization)
        if len(image.shape) == 2:
            # Add batch and channel dimensions: (28, 28) -> (1, 28, 28, 1)
            model_input = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            # Add batch dimension: (28, 28, 1) -> (1, 28, 28, 1)
            model_input = image.reshape(1, 28, 28, 1)
        else:
            model_input = image
        
        # Ensure float32 and proper normalization
        model_input = model_input.astype('float32')
        if np.max(model_input) > 1:
            model_input = model_input / 255.0
        
        # Make prediction using the Keras model directly
        probabilities = model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        # Create probability distribution for all digits
        probabilities_dict = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        
        # Generate confidence analysis by modifying parts of the image
        confidence_analysis = generate_confidence_analysis(image, model)
        
        prediction_result = {
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities_dict,
            'confidence_analysis': confidence_analysis,
            'image_stats': {
                'mean_intensity': float(np.mean(image)),
                'std_intensity': float(np.std(image)),
                'non_zero_pixels': int(np.count_nonzero(image)),
                'total_pixels': int(image.size)
            }
        }
        
        logger.info(f"Prediction completed: digit {predicted_digit} with confidence {confidence:.3f}")
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise RuntimeError(f"Prediction failed: {e}")

def generate_confidence_analysis(image, model):
    """Generate confidence analysis by testing different noise levels"""
    logger.info("Generating confidence analysis")
    
    try:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        analysis = []
        
        for noise_level in noise_levels:
            # Add gaussian noise to the image
            noisy_image = image.copy()
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, image.shape)
                noisy_image = np.clip(noisy_image + noise, 0, 1)
            
            # Prepare noisy image for model prediction
            if len(noisy_image.shape) == 2:
                model_input = noisy_image.reshape(1, 28, 28, 1)
            else:
                model_input = noisy_image.reshape(1, 28, 28, 1)
            
            model_input = model_input.astype('float32')
            if np.max(model_input) > 1:
                model_input = model_input / 255.0
            
            # Make prediction on noisy image
            probabilities = model.predict(model_input, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
            analysis.append({
                'noise_level': float(noise_level),
                'confidence': confidence,
                'predicted_digit': predicted_digit
            })
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Could not generate confidence analysis: {e}")
        return []

def output_fn(predictions, original_data):
    """Format output for MNIST classification results"""
    logger.info(f"Formatting output for predictions: digit {predictions['digit']}")
    
    # Create section to show original and processed images
    
    image_section = {
        "type": "section",
        "id": "image_comparison",
        "title": "Original vs Processed Images",
        "items_per_row": 2,
        "items": [
            {
                "type": "image",
                "id": "original_image",
                "title": "Original Image",
                "description": "The original image submitted for classification.",
                "file_path": original_data['number_image']['path']
            },
            {
                "type": "image",
                "id": "processed_image",
                "title": "Processed Image",
                "description": "The processed image used for classification.",
                "file_path": os.path.join(os.path.dirname(__file__), f"processed_images/{original_data['number_image']['name']}_processed_28x28.png")
            }
        ]
    }

    # Create results section with MNIST classification information
    results_section = {
        "type": "section",
        "id": "mnist_results",
        "title": "MNIST Digit Classification Results",
        "color": "blue",
        "description": "Model prediction for handwritten digit classification.",
        "items": [
            {
                "type": "table",
                "id": "classification_results",
                "title": "Classification Results",
                "data": {
                    "columns": [
                        {"header": "Metric", "field": "metric"},
                        {"header": "Value", "field": "value"}
                    ],
                    "rows": [
                        {
                            "metric": "Predicted Digit",
                            "value": str(predictions['digit'])
                        },
                        {
                            "metric": "Confidence Score",
                            "value": f"{predictions['confidence']:.3f}"
                        },
                        {
                            "metric": "Non-zero Pixels",
                            "value": f"{predictions['image_stats']['non_zero_pixels']}/784"
                        },
                        {
                            "metric": "Mean Intensity",
                            "value": f"{predictions['image_stats']['mean_intensity']:.3f}"
                        }
                    ]
                }
            }
        ]
    }
    
    # Create probability distribution chart
    probability_section = {
        "type": "section",
        "id": "probability_distribution",
        "title": "Probability Distribution",
        "description": "Confidence scores for each digit class.",
        "items": [
            {
                "type": "bar_chart",
                "id": "digit_probabilities",
                "title": "Digit Probability Distribution",
                "description": "Probability distribution across all digit classes.",
                "data": {
                    "bars": [
                        {
                            "id": digit,
                            "value": round(predictions['probabilities'].get(digit, 0.0), 3),
                            "label": f"Digit {digit}",
                            "style": {
                                "color": DIGIT_COLORS[int(digit)] if digit.isdigit() else "#cccccc"
                            }
                        }
                        for digit in DIGIT_CLASSES
                    ],
                    "axis": {
                        "x": {"label": "Digit"},
                        "y": {"label": "Probability"}
                    }
                }
            }
        ]
    }
    
    # Create confidence analysis section if available
    confidence_analysis = predictions.get('confidence_analysis', [])
    if confidence_analysis:
        confidence_section = {
            "type": "section",
            "id": "confidence_analysis",
            "title": "Robustness Analysis",
            "description": "How prediction confidence changes with image noise.",
            "items": [
                {
                    "type": "line_chart",
                    "id": "noise_sensitivity",
                    "title": "Confidence vs Noise Level",
                    "description": "Model confidence as noise is added to the image.",
                    "data": {
                        "lines": [
                            {
                                "id": "confidence",
                                "points": [
                                    {
                                        "x": round(point['noise_level'], 2),
                                        "y": round(point['confidence'], 3)
                                    }
                                    for point in confidence_analysis
                                ],
                                "style": {
                                    "color": "#1f77b4",
                                    "width": 2
                                }
                            }
                        ],
                        "axis": {
                            "x": {"label": "Noise Level"},
                            "y": {"label": "Confidence"}
                        }
                    }
                }
            ]
        }
        
        response = [image_section, results_section, probability_section, confidence_section]
    else:
        response = [image_section, results_section, probability_section]

    logger.info(f"Output formatted with {len(response)} sections")
    return response