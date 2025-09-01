import json
import logging
import os

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

def model_fn(model_dir):
    logger.info(f"Loading model from directory: {model_dir}")
    return 1

def input_fn(request_data):
    logger.info(f"Processing request_data: {request_data}")
    with open(request_data['input_file']['path'], 'r') as f:
        input_data = json.load(f)
    logger.info(f"Loaded input_data: {input_data}")
    return input_data

def predict_fn(input_data, model):
    logger.info(f"Making prediction with input_data: {input_data}, model: {model}")
    return input_data

def output_fn(predictions, original_data):
    logger.info(f"Formatting output for predictions: {predictions}")
    
    # Add section for a table displaying predictions in Celsius and Fahrenheit
    results_section = {
        "type": "section",
        "id": "results",
        "title": "Prediction Results",
        "color": "green",  # Optional: color for section header
        "description": "Model prediction for temperature in Celsius and Fahrenheit.",
        "items": [
            {
                "type": "table",
                "id": "prediction_table",
                "title": "Prediction Summary",
                "data": {
                    "columns": [
                        {"header": "Input", "field": "input"},
                        {"header": "Value", "field": "value"}
                    ],
                    "rows": [
                        {
                            "input": "Number",
                            "value": f"{original_data['input_value']}"
                        },
                        {
                            "input": "File",
                            "value": f"{original_data['input_file']}"
                        },
                        # # if multiple is true, the output is a list (for multiple files)
                        # {
                        #     "input": "File name",
                        #     "value": f"{original_data['input_file'][0]['name']}"
                        # }
                        # if multiple is false, the output is a single value (file)
                        {
                            "input": "File name",
                            "value": f"{original_data['input_file']['name']}"
                        },
                        # input data from input_example.json
                        {
                            "input": "Skills",
                            "value": ", ".join(predictions['skills'])
                        }
                    ]
                }
            },
            {
                "type": "image",
                "id": "image1",
                "title": "Example Image",
                "description": "An image example", 
                "caption": "This is an example image", 
                "comment": "This image illustrates the concept",
                "file_path": "actualvspred_cpdp.png"
            }
        ]
    }

    response = [results_section]

    return response