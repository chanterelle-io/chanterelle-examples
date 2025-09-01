import json
import logging
import os
import sys
import joblib
import numpy as np

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

# Define feature names for California housing
FEATURE_NAMES = [
    'median_income', 'house_age', 'avg_rooms', 'avg_bedrooms', 
    'population', 'avg_occupancy', 'latitude', 'longitude'
]

# Define price categories
def get_price_category(price):
    """Convert price to category"""
    if price < 2.0:  # Less than $200k
        return "Low"
    elif price < 4.0:  # $200k - $400k
        return "Medium"
    else:  # Above $400k
        return "High"

# Define the ranges for each feature for sensitivity analysis
INPUT_RANGES = {
    'Median Income (tens of thousands)': {'input_number': 0, 'range': np.arange(0.5, 15.0, 0.5), 'min': 0.5, 'max': 15.0},
    'House Age (years)': {'input_number': 1, 'range': np.arange(1, 52, 3), 'min': 1, 'max': 52},
    'Average Rooms per Household': {'input_number': 2, 'range': np.arange(3.0, 10.0, 1), 'min': 3.0, 'max': 10.0},
    'Latitude': {'input_number': 6, 'range': np.arange(32.5, 42.0, 0.5), 'min': 32.5, 'max': 42.0},
}

def model_fn(model_dir):
    """Load the California housing regression model from the specified directory"""
    logger.info(f"Loading California housing regression model from directory: {model_dir}")
    try:
        model_path = os.path.join(model_dir, 'california_housing_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")
    
def input_fn(request_data):
    """Extract housing features from request data"""
    logger.info(f"Processing request_data: {request_data}")
    try:
        # Extract the eight housing features from the request
        median_income = request_data.get('median_income')
        house_age = request_data.get('house_age')
        avg_rooms = request_data.get('avg_rooms')
        avg_bedrooms = request_data.get('avg_bedrooms')
        population = request_data.get('population')
        avg_occupancy = request_data.get('avg_occupancy')
        latitude = request_data.get('latitude')
        longitude = request_data.get('longitude')

        # Check that all values are provided
        required_features = [median_income, house_age, avg_rooms, avg_bedrooms, 
                           population, avg_occupancy, latitude, longitude]
        if any(val is None for val in required_features):
            raise ValueError("Missing one or more required housing features")

        # Create feature array for the model
        features = np.array([[median_income, house_age, avg_rooms, avg_bedrooms,
                            population, avg_occupancy, latitude, longitude]])
        
        input_data = {
            'features': features
        }
        logger.info(f"Processed input_data with shape: {features.shape}")
        return input_data
    except Exception as e:
        logger.error(f"Error processing input_data: {e}")
        raise ValueError(f"Could not process input data: {e}")

def predict_fn(input_data, model):
    """Make prediction using the California housing regression model"""
    logger.info(f"Making prediction with input_data: {input_data['features'].shape}")    
    # Get features array from input data
    features = input_data['features']
    
    # Make prediction
    predicted_price = model.predict(features)[0]
    price_category = get_price_category(predicted_price)
    
    # Sensitivity curves
    # format {input_column: [{'x': val, 'y': predicted_price}]}
    sensitivity_curves_1d = {}
    for col in ['Median Income (tens of thousands)', 'House Age (years)', 
                'Average Rooms per Household', 'Latitude']:
        sensitivity_curves_1d[col] = []
        input_config = INPUT_RANGES[col]
        for x in input_config['range']:
            fts = features.copy()
            fts[0, input_config['input_number']] = x
            price_pred = model.predict(fts)[0]
            sensitivity_curves_1d[col].append({'x': float(x), 'y': float(price_pred)})

    result = {
        'predicted_price': float(predicted_price),
        'price_category': price_category,
        'price_formatted': f"${predicted_price:.2f}k (${predicted_price*100:.0f})",
        'sensitivity_curves': sensitivity_curves_1d
    }
    logger.info(f"Prediction completed: ${predicted_price:.2f}k ({price_category})")
    return result
    

def output_fn(predictions, original_data):
    """Format output for California housing price prediction results"""
    logger.info(f"Formatting output for predictions: {predictions}")
    
    # Create results section with housing price prediction information
    results_section = {
        "type": "section",
        "id": "housing_results",
        "title": "California Housing Price Prediction",
        "color": "blue",
        "description": "Model prediction for median house value in California.",
        "items": [
            {
                "type": "table",
                "id": "prediction_results",
                "title": "Prediction Results",
                "data": {
                    "columns": [
                        {"header": "Metric", "field": "metric"},
                        {"header": "Value", "field": "value"}
                    ],
                    "rows": [
                        {
                            "metric": "Predicted Price",
                            "value": predictions['price_formatted']
                        },
                        {
                            "metric": "Price Category",
                            "value": predictions['price_category']
                        },
                        {
                            "metric": "Price (hundreds of thousands)",
                            "value": f"{predictions['predicted_price']:.2f}"
                        }
                    ]
                }
            }
        ]
    }
    
    sensitivity_section = {
        "type": "section",
        "id": "sensitivity_curves",
        "title": "Sensitivity Analysis",
        "description": "How changes in key features affect the predicted house price.",
        "items_per_row": 2,
        "items": []
    }
    
    # Sensitivity curves
    # format {input_column: [{'x': val, 'y': predicted_price}]}
    sensitivity_curves_1d = predictions.get('sensitivity_curves', {})
    curve_colors = ['blue', 'red', 'green', 'orange']
    
    for i, (col, curve_data) in enumerate(sensitivity_curves_1d.items()):
        logger.info(f"Processing sensitivity curves for {col}")
        color = curve_colors[i % len(curve_colors)]
        
        # Use bar chart for Average Rooms per Household, line chart for others
        if col == "Average Rooms per Household":
            item = {
                "type": "bar_chart",
                "id": f"sensitivity_curve_{col.replace(' ', '_').replace('(', '').replace(')', '').lower()}",
                "title": f"Price vs {col}",
                "description": f"How {col.lower()} affects predicted house price.",
                "data": {
                    "bars": [
                        {
                            "label": f"{int(point['x'])} rooms",
                            "value": round(point['y'], 2),
                            "color": color
                        }
                        for point in curve_data
                    ],
                    "axis": {
                        "x": {"label": col},
                        "y": {"label": "Predicted Price (hundreds of thousands $)"}
                    },
                    "orientation": "vertical"
                }
            }
        else:
            item = {
                "type": "line_chart",
                "id": f"sensitivity_curve_{col.replace(' ', '_').replace('(', '').replace(')', '').lower()}",
                "title": f"Price vs {col}",
                "description": f"How {col.lower()} affects predicted house price.",
                "data": {
                    "lines": [
                        {
                            "id": "price_sensitivity",
                            "points": [
                                {"x": round(point['x'], 2), "y": round(point['y'], 2)}
                                for point in curve_data
                            ],
                            "style": {
                                "color": color,
                                "width": 2
                            }
                        }
                    ],
                    "axis": {
                        "x": {"label": col},
                        "y": {"label": "Predicted Price (hundreds of thousands $)"}
                    }
                }
            }
        sensitivity_section["items"].append(item)

    response = [results_section, sensitivity_section]
    return response