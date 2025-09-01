import json
import logging
import os
import sys
import joblib
import numpy as np
import base64
from io import BytesIO
try:
    from PIL import Image
except ImportError:
    Image = None

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

# Define digit classes and colors for visualization
DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
DIGIT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def model_fn(model_dir):
    """Load the MNIST classification model from the specified directory"""
    logger.info(f"Loading MNIST classification model from directory: {model_dir}")
    try:
        # Try to load the wrapper first (preferred)
        wrapper_path = os.path.join(model_dir, 'mnist_classification_wrapper.joblib')
        if os.path.exists(wrapper_path):
            model = joblib.load(wrapper_path)
            logger.info("Wrapper model loaded successfully")
            return model
        
        # Fallback to TensorFlow model
        model_path = os.path.join(model_dir, 'mnist_classification_model.h5')
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                keras_model = tf.keras.models.load_model(model_path)
                
                # Create a simple wrapper if TensorFlow model is loaded
                class SimpleWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict(self, image):
                        if len(image.shape) == 2:
                            image = image.reshape(1, 28, 28, 1)
                        elif len(image.shape) == 3:
                            image = image.reshape(1, 28, 28, 1)
                        
                        if np.max(image) > 1:
                            image = image.astype('float32') / 255.0
                        
                        probabilities = self.model.predict(image, verbose=0)[0]
                        predicted_digit = np.argmax(probabilities)
                        confidence = float(np.max(probabilities))
                        
                        return {
                            'digit': int(predicted_digit),
                            'confidence': confidence,
                            'all_probabilities': probabilities.tolist()
                        }
                
                model = SimpleWrapper(keras_model)
                logger.info("TensorFlow model loaded and wrapped successfully")
                return model
            except ImportError:
                raise RuntimeError("TensorFlow not available and no wrapper model found")
        
        raise FileNotFoundError(f"No model file found in: {model_dir}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")

def input_fn(request_data):
    """Extract image data from request - primarily expects file paths to image files"""
    logger.info(f"Processing request_data with keys: {list(request_data.keys()) if isinstance(request_data, dict) else 'not a dict'}")
    try:
        image_data = None
        
        # Primary format: file path to an image
        if 'image_path' in request_data or 'file_path' in request_data or 'path' in request_data:
            image_path = request_data.get('image_path') or request_data.get('file_path') or request_data.get('path')
            
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            if Image is None:
                raise ValueError("PIL not available for image processing")
            
            try:
                # Load image using PIL
                logger.info(f"Loading image from file: {image_path}")
                pil_image = Image.open(image_path)
                
                # Convert to grayscale and resize to 28x28
                pil_image = pil_image.convert('L').resize((28, 28))
                image_data = np.array(pil_image)
                
                logger.info(f"Loaded image from {image_path}, original mode: {Image.open(image_path).mode}")
                
            except Exception as e:
                logger.error(f"Failed to load image from {image_path}: {e}")
                raise ValueError(f"Could not load image file: {e}")
        
        # Alternative: if 'image' contains a file path (string that looks like a path)
        elif 'image' in request_data:
            image_input = request_data['image']
            
            # Check if it's a file path
            if isinstance(image_input, str) and (
                image_input.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')) or
                os.path.exists(image_input)
            ):
                if not os.path.exists(image_input):
                    raise ValueError(f"Image file not found: {image_input}")
                
                if Image is None:
                    raise ValueError("PIL not available for image processing")
                
                try:
                    logger.info(f"Loading image from file: {image_input}")
                    pil_image = Image.open(image_input)
                    pil_image = pil_image.convert('L').resize((28, 28))
                    image_data = np.array(pil_image)
                except Exception as e:
                    logger.error(f"Failed to load image from {image_input}: {e}")
                    raise ValueError(f"Could not load image file: {e}")
            
            # Handle base64 encoded image
            elif isinstance(image_input, str) and not os.path.exists(image_input):
                try:
                    # Remove data URL prefix if present
                    if image_input.startswith('data:image'):
                        image_input = image_input.split(',')[1]
                    
                    # Decode base64
                    image_bytes = base64.b64decode(image_input)
                    
                    if Image is not None:
                        # Use PIL if available
                        pil_image = Image.open(BytesIO(image_bytes))
                        # Convert to grayscale and resize to 28x28
                        pil_image = pil_image.convert('L').resize((28, 28))
                        image_data = np.array(pil_image)
                    else:
                        raise ValueError("PIL not available for image processing")
                        
                except Exception as e:
                    logger.warning(f"Failed to decode base64 image: {e}")
                    raise ValueError(f"Invalid base64 image data: {e}")
            
            # Handle numpy array or list
            elif isinstance(image_input, (list, np.ndarray)):
                image_data = np.array(image_input)
                
                # Ensure it's 28x28
                if image_data.shape != (28, 28):
                    if image_data.size == 784:  # Flattened 28x28
                        image_data = image_data.reshape(28, 28)
                    else:
                        raise ValueError(f"Image must be 28x28, got shape: {image_data.shape}")
        
        # Handle raw pixel data
        elif 'pixels' in request_data:
            pixels = request_data['pixels']
            if len(pixels) == 784:
                image_data = np.array(pixels).reshape(28, 28)
            else:
                raise ValueError(f"Pixels array must have 784 elements, got {len(pixels)}")
        
        # Handle individual pixel values (for testing)
        elif all(f'pixel_{i}' in request_data for i in range(784)):
            pixels = [request_data[f'pixel_{i}'] for i in range(784)]
            image_data = np.array(pixels).reshape(28, 28)
        
        else:
            raise ValueError("No valid image data found. Expected 'image_path'/'file_path'/'path' (file path), 'image' (file path, base64, or array), 'pixels', or 'pixel_0' through 'pixel_783'")
        
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
        
    except Exception as e:
        logger.error(f"Error processing input_data: {e}")
        raise ValueError(f"Could not process input data: {e}")

def predict_fn(input_data, model):
    """Make prediction using the MNIST classification model"""
    logger.info(f"Making prediction with image shape: {input_data['image'].shape}")
    
    try:
        # Get image from input data
        image = input_data['image']
        
        # Make prediction using the model
        result = model.predict(image)
        
        # Extract prediction results
        predicted_digit = result['digit']
        confidence = result['confidence']
        all_probabilities = result.get('all_probabilities', [])
        
        # Create probability distribution for all digits
        probabilities_dict = {str(i): float(prob) for i, prob in enumerate(all_probabilities)}
        
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
            
            # Make prediction on noisy image
            result = model.predict(noisy_image)
            
            analysis.append({
                'noise_level': float(noise_level),
                'confidence': float(result['confidence']),
                'predicted_digit': int(result['digit'])
            })
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Could not generate confidence analysis: {e}")
        return []

def output_fn(predictions, original_data):
    """Format output for MNIST classification results"""
    logger.info(f"Formatting output for predictions: digit {predictions['digit']}")
    
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
        
        response = [results_section, probability_section, confidence_section]
    else:
        response = [results_section, probability_section]
    
    logger.info(f"Output formatted with {len(response)} sections")
    return response