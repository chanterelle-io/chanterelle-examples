import sys
import os
import joblib
import numpy as np


# Define iris species names
IRIS_SPECIES = ['setosa', 'versicolor', 'virginica']

def model_fn(model_dir):
    """Load the iris classification model from the specified directory"""
    model_path = os.path.join(model_dir, 'iris_classification_model.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(request_data):
    """Extract iris flower measurements from request data"""
    # Extract the four iris measurements from the request
    sepal_length = request_data['sepal_length']
    sepal_width = request_data['sepal_width']
    petal_length = request_data['petal_length']
    petal_width = request_data['petal_width']

    # Create feature array for the model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    input_data = {
        'features': features
    }
    
    return input_data

def predict_fn(input_data, model):
    """Make prediction using the iris classification model"""    
    try:
        # Get features array from input data
        features = input_data['features']
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get species name and confidence
        species = IRIS_SPECIES[prediction]
        confidence = float(np.max(probabilities))
        
        result = {
            'species': species,
            'confidence': confidence
        }
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not load some resources: {e}", file=sys.stderr, flush=True) 

def output_fn(predictions, original_data):
    """Format output for iris classification results"""   
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
                            "value": predictions.get('species', 'Unknown')
                        },
                        {
                            "metric": "Confidence Score",
                            "value": f"{predictions.get('confidence', 0):.3f}"
                        }
                    ]
                }
            }
        ]
    }

    response = [results_section]
    return response