import json
import logging
import os
import sys
from typing import Dict, List, Any, Union, TypedDict, Protocol
import joblib
import numpy as np
from numpy.typing import NDArray

"""
Expected Input Schema:
{
    "sepal_length": float,  # Sepal length in cm (4.3-7.9)
    "sepal_width": float,   # Sepal width in cm (2.0-4.4) 
    "petal_length": float,  # Petal length in cm (1.0-6.9)
    "petal_width": float    # Petal width in cm (0.1-2.5)
}
"""

# Minimal types to document interfaces (PEP 484 / TypedDict / Protocol)
class RequestData(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class FeaturesDict(TypedDict):
    # Single-sample feature matrix with shape (1, 4)
    features: NDArray[np.float64]


class ClassifierProtocol(Protocol):
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]: ...
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...

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

# Define iris species names
IRIS_SPECIES = ['setosa', 'versicolor', 'virginica']
CURVE_COLORS = ['blue', 'orange', 'green']

# Define the ranges for each feature for sensitivity analysis
INPUT_RANGES = {
    'sepal length (cm)': {'input_number': 0, 'range': np.arange(4.3, 7.9, 0.5), 'min': 4.3, 'max': 7.9},
    'sepal width (cm)': {'input_number': 1, 'range': np.arange(2.0, 4.4, 0.5), 'min': 2.0, 'max': 4.4},
    'petal length (cm)': {'input_number': 2, 'range': np.arange(1.0, 6.9, 0.5), 'min': 1.0, 'max': 6.9},
    'petal width (cm)': {'input_number': 3, 'range': np.arange(0.1, 2.5, 0.1), 'min': 0.1, 'max': 2.5},
}

def model_fn(model_dir: str) -> Any:
    """Load the iris classification model from the specified directory"""
    logger.info(f"Loading iris classification model from directory: {model_dir}")
    try:
        model_path = os.path.join(model_dir, 'iris_classification_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")
    
def input_fn(request_data: RequestData) -> FeaturesDict:
    """Extract iris flower measurements from request data
    
    Args:
        request_data (dict): Input data containing:
            - sepal_length (float): Sepal length in cm
            - sepal_width (float): Sepal width in cm  
            - petal_length (float): Petal length in cm
            - petal_width (float): Petal width in cm
            
    Returns:
        dict: Contains 'features' key with numpy array of shape (1, 4)
        
    Raises:
        ValueError: If any required measurements are missing
    """
    logger.info(f"Processing request_data: {request_data}")
    try:
        # Extract and coerce the four iris measurements from the request
        sepal_length = float(request_data['sepal_length'])
        sepal_width = float(request_data['sepal_width'])
        petal_length = float(request_data['petal_length'])
        petal_width = float(request_data['petal_width'])

        # Create feature array for the model (ensure float64 dtype)
        features: NDArray[np.float64] = np.array(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            dtype=np.float64,
        )

        input_data: FeaturesDict = {'features': features}
        logger.info(f"Processed input_data with shape: {features.shape}")
        return input_data
    except KeyError as e:
        logger.error(f"Missing required measurement: {e}")
        raise ValueError(f"Missing required measurement: {e}")
    except Exception as e:
        logger.error(f"Error processing input_data: {e}")
        raise ValueError(f"Could not process input data: {e}")

def predict_fn(input_data: FeaturesDict, model: ClassifierProtocol) -> Dict[str, Any]:
    """Make prediction using the iris classification model
    
    Args:
        input_data: Dictionary containing 'features' numpy array
        model: Trained scikit-learn model
        
    Returns:
        dict: Prediction results containing species, confidence, probabilities, and sensitivity curves
    """
    logger.info(f"Making prediction with input_data: {input_data['features'].shape}")    
    # Get features array from input data
    features = input_data['features']
    
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Get species name and confidence
    species = IRIS_SPECIES[prediction]
    confidence = float(np.max(probabilities))
    # Sensitivity curves
    # format {input_column: {species: [{'x': val, 'y': probability}]}}
    sensitivity_curves_1d = {}
    for col in ['petal length (cm)', 'petal width (cm)']:
        sensitivity_curves_1d[col] = {s: [] for s in IRIS_SPECIES}
        input_config = INPUT_RANGES[col]
        for x in input_config['range']:
            fts = features.copy()
            fts[0, input_config['input_number']] = x
            for s, prob in zip(IRIS_SPECIES, model.predict_proba(fts)[0]):
                sensitivity_curves_1d[col][s].append({'x': float(x), 'y': float(prob)})

    result = {
        'species': species,
        'confidence': confidence,
        'probabilities': {species: float(prob) for species, prob in zip(IRIS_SPECIES, probabilities)},
        'sensitivity_curves': sensitivity_curves_1d
    }
    logger.info(f"Prediction completed: {species} with confidence {confidence:.3f}")
    return result
    

def output_fn(predictions: Dict[str, Any], original_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format output for iris classification results
    
    Args:
        predictions: Dictionary containing model predictions and analysis
        original_data: Original input data (unused but kept for interface compatibility)
        
    Returns:
        list: Formatted sections for display containing results and sensitivity curves
    """
    logger.info(f"Formatting output for predictions: {predictions}")
    
    # Create results section with iris classification information
    results_section = {
        "type": "section",
        "id": "iris_results",
        "title": "Iris Classification Results",
        "color": "green",
        "description": "Model prediction for iris flower species classification.",
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
                            "metric": "Predicted Species",
                            "value": predictions['species']
                        },
                        {
                            "metric": "Confidence Score",
                            "value": f"{predictions['confidence']:.3f}"
                        }
                    ]
                }
            }
        ]
    }
    sensitivity_section = {
        "type": "section",
        "id": "sensitivity_curves",
        "title": "Sensitivity Curves",
        "description": "Sensitivity curves for the model's predictions.",
        "items_per_row": 2,
        "items": []
    }
    # Sensitivity curves
    # format {input_column: {species: [{'x': val, 'y': probability}]}}
    sensitivity_curves_1d = predictions.get('sensitivity_curves', {})
    for col, curves in sensitivity_curves_1d.items():
        logger.info(f"Processing sensitivity curves for {col}")
        item= {
            "type": "line_chart",
            "id": f"sensitivity_curve_{col}",
            "title": f"Sensitivity Curve for {col}",
            "description": f"Sensitivity curve for {col} feature.",
            "data": {
                "lines": [],
                "axis": {
                    "x": {"label": col},
                    "y": {"label": "Probability"}
                }
            }
        }
        for s, color in zip(IRIS_SPECIES, CURVE_COLORS):
            line = {
                "id": s,
                "points": [
                    {"x": round(point['x'], 2), "y": round(point['y'], 2)}
                    for point in curves.get(s, [])
                ],
                "style": {
                    "color": color,
                    "width": 2
                }
            }
            item["data"]["lines"].append(line)
        sensitivity_section["items"].append(item)


    response = [results_section, sensitivity_section]
    return response