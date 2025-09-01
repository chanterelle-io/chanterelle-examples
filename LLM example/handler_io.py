import anthropic
import logging
import os

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic(api_key=api_key)

# Define the categorization tool
tools = [
    {
        "name": "categorize_interruption",
        "description": "Categorize a machine interruption statement from a galvanizing line",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["laitevika", "laatuongelma", "materiaaliongelma", "suunniteltu_huolto", "kayttajavirhe"],
                    "description": "The category of the interruption"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score for the categorization"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the categorization in Finnish"
                }
            },
            "required": ["category", "confidence", "reasoning"]
        }
    }
]

def categorize_statement(client,statement):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        tools=tools,
        tool_choice={"type": "tool", "name": "categorize_interruption"},
        messages=[
            {
                "role": "user",
                "content": f"""Analysoi seuraava galvanointilinjan keskeytystilanne ja luokittele se yhteen seuraavista kategorioista:

                - laitevika: Laitteiden rikkoutumiset, tekniset viat
                - laatuongelma: Pinnoitteen laatu, paksuus, tarttuvuus
                - materiaaliongelma: Kemikaalit, sinkkitaso, kappaleiden kunto  
                - suunniteltu_huolto: Määräaikaishuollot, puhdistukset
                - kayttajavirhe: Inhimilliset virheet, väärät asetukset

                Tilanne: "{statement}"

                Käytä categorize_interruption-työkalua vastauksessasi."""
            }
        ]
    )
    
    return message.content[0].input

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
    client = anthropic.Anthropic(api_key=api_key)
    return client

def input_fn(request_data):
    """Extract text input from request data
    
    Args:
        request_data (dict): Input data containing:
            - text_input (str): Text description of the interruption
            
    Returns:
        dict: Contains 'text_input' key with the input string
        
    Raises:
        ValueError: If text_input is missing
    """
    logger.info(f"Processing request_data: {request_data}")
    try:
        text_input = str(request_data['text_input'])
        return {'text_input': text_input}
    except KeyError as e:
        logger.error(f"Missing required input: {e}")
        raise ValueError(f"Missing required input: {e}")
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise ValueError(f"Error processing input: {e}")

def predict_fn(input_data, model):
    """Make a prediction using the model

    Args:
        input_data (dict): The input data for the prediction
        model: The model to use for prediction

    Returns:
        dict: The prediction result
    """
    logger.info(f"Making prediction for input_data: {input_data}")
    try:
        response = categorize_statement(model, input_data['text_input'])
        return response
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise ValueError(f"Error making prediction: {e}")

def output_fn(predictions, original_data):
    """Format the prediction output
    
    Args:
        predictions (dict): The prediction result from the model
        original_data (dict): The original input data for reference
        
    Returns:
        dict: Formatted output containing category, confidence, reasoning, and original input
    """
    logger.info(f"Formatting output for predictions: {predictions}")
    try:
        output = {
            'category': predictions['category'],
            'confidence': predictions['confidence'],
            'reasoning': predictions['reasoning'],
            'original_input': original_data['text_input']
        }

        results_section = {
        "type": "section",
        "id": "interruption_category",
        "title": "Interruption Category",
        "color": "blue",
        "description": "Model prediction for interruption categorization.",
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
                            "metric": "Predicted Category",
                            "value": predictions['category']
                        },
                        {
                            "metric": "Confidence Score",
                            "value": f"{predictions['confidence']:.3f}"
                        },
                        {
                            "metric": "Reasoning",
                            "value": predictions['reasoning']
                        }
                    ]
                }
            }
        ]
    }

        return [results_section]
    except KeyError as e:
        logger.error(f"Missing expected prediction key: {e}")
        raise ValueError(f"Missing expected prediction key: {e}")
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        raise ValueError(f"Error formatting output: {e}")

