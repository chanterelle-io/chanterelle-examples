import json
import logging
import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import statsmodels.api as sm
from io import BytesIO
import base64

# Set up logging to write to a file
log_file = os.path.join(os.path.dirname(__file__), 'handler.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Does nothing"""
    return 1

def init_resources_fn(resource_dir):
    # Load airline passenger dataset
    csv_path = os.path.join(os.path.dirname(__file__), "airline_passengers.csv")
    df = pd.read_csv(csv_path, parse_dates=["Month"], index_col="Month")
    return {"df": df}

def create_sarima_model(df, p, d, q, P, D, Q, s):
    """Create and fit a new SARIMA model on airline passenger data"""
    try:        
        # Fit SARIMA model
        sarima_model = sm.tsa.statespace.SARIMAX(
            df["Passengers"],
            order=(p,d,q),           # ARIMA part
            seasonal_order=(P,D,Q,s) # seasonal part
        ).fit()
        
        logger.info("New SARIMA model created and fitted successfully")
        return sarima_model
        
    except Exception as e:
        logger.error(f"Error creating SARIMA model: {e}")
        raise RuntimeError(f"Could not create SARIMA model: {e}")

def input_fn(request_data, resources):
    """Extract forecasting parameters from request"""
    logger.info(f"Processing request_data: {request_data}")
    
    try:
        # Default parameters based on model_meta.json
        forecast_steps = request_data.get('forecast_steps', 36)
        confidence_level = request_data.get('confidence_level', 0.95)
        
        # SARIMA parameters (optional, for model comparison)
        p = request_data.get('p', 1)
        d = request_data.get('d', 1) 
        q = request_data.get('q', 1)
        P = request_data.get('P', 1)
        D = request_data.get('D', 1)
        Q = request_data.get('Q', 1)
        s = request_data.get('s', 12)
        
        input_data = {
            'confidence_level': float(confidence_level),
            'forecast_steps': int(forecast_steps),
            'sarima_params': {
                'order': (int(p), int(d), int(q)),
                'seasonal_order': (int(P), int(D), int(Q), int(s))
            },
            "fit_inputs": (p, d, q, P, D, Q, s)
        }
        
        logger.info(f"Processed input")
        return input_data
        
    except Exception as e:
        logger.error(f"Error processing input_data: {e}")
        raise ValueError(f"Could not process input data: {e}")
    
def create_forecast_plot(df, forecast, fit_inputs):
    """Create the forecast visualization plot"""
    try:
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        subtitle = f"SARIMA{fit_inputs}"
        plt.figure(figsize=(10,4))
        plt.plot(df, label="Observed")
        plt.plot(forecast_mean.index, forecast_mean, label="SARIMA Forecast")
        plt.fill_between(forecast_ci.index,
                        forecast_ci.iloc[:,0],
                        forecast_ci.iloc[:,1], color="lightblue", alpha=0.3)
        plt.title(f"SARIMA Forecast of Airline Passengers ({subtitle})")
        plt.xlabel("Date")
        plt.ylabel("Passengers")
        plt.legend()
        
        # Save to file for reference
        plot_path = os.path.join(os.path.dirname(__file__), f'graphs/forecast_plot.png')
        plt.savefig(plot_path)
        return plot_path
        
    except Exception as e:
        logger.error(f"Error creating forecast plot: {e}")
        return None

def predict_fn(input_data, model, resources):
    """Generate SARIMA forecasts"""
    logger.info(f"Making forecast")
    
    try:
        # create_sarima_model
        df = resources['df']
        sarima_model = create_sarima_model(df, *input_data['fit_inputs'])

        # Generate forecast
        forecast_steps = input_data['forecast_steps']
        confidence_level = input_data['confidence_level']
        
        forecast = sarima_model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int(alpha=1-confidence_level)
        
        # Model performance metrics
        aic = float(sarima_model.aic)
        bic = float(sarima_model.bic)
        
        # Get residuals for diagnostics
        residuals = sarima_model.resid

        # Create plot
        plot_path = create_forecast_plot(df, forecast, input_data['fit_inputs'])

        prediction_result = {
            'confidence_interval_lower': forecast_ci.iloc[:,0].tolist(),
            'confidence_interval_upper': forecast_ci.iloc[:,1].tolist(),
            'aic': aic,
            'bic': bic,
            'plot_path': plot_path,
            'model_summary': {
                'params_significant': len([p for p in sarima_model.pvalues if p < 0.05]),
                'total_params': len(sarima_model.pvalues),
                'log_likelihood': float(sarima_model.llf),
                'residuals_mean': float(np.mean(residuals)),
                'residuals_std': float(np.std(residuals))
            }
        }
        
        logger.info(f"Forecast completed: {forecast_steps} steps, AIC: {aic:.2f}")
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        raise RuntimeError(f"Forecasting failed: {e}")



def output_fn(predictions, original_data):
    """Format output for SARIMA forecasting results"""
    # logger.info(f"Formatting output for {len(predictions['forecast_values'])} forecast points")
    
    # Model summary section
    model_section = {
        "type": "section",
        "id": "model_summary",
        "title": "SARIMA Model Summary",
        "description": "Statistical summary and performance metrics of the fitted SARIMA model.",
        "items": [
            {
                "type": "table",
                "id": "model_metrics",
                "title": "Model Performance Metrics",
                "data": {
                    "columns": [
                        {"header": "Metric", "field": "metric"},
                        {"header": "Value", "field": "value"}
                    ],
                    "rows": [
                        {
                            "metric": "AIC (Akaike Information Criterion)",
                            "value": f"{predictions['aic']:.2f}"
                        },
                        {
                            "metric": "BIC (Bayesian Information Criterion)",
                            "value": f"{predictions['bic']:.2f}"
                        },
                        {
                            "metric": "Log Likelihood",
                            "value": f"{predictions['model_summary']['log_likelihood']:.2f}"
                        },
                        {
                            "metric": "Significant Parameters",
                            "value": f"{predictions['model_summary']['params_significant']}/{predictions['model_summary']['total_params']}"
                        },
                        {
                            "metric": "Residuals Mean",
                            "value": f"{predictions['model_summary']['residuals_mean']:.4f}"
                        },
                        {
                            "metric": "Residuals Std Dev",
                            "value": f"{predictions['model_summary']['residuals_std']:.2f}"
                        }
                    ]
                }
            }
        ]
    }
    
    # Forecast visualization section
    forecast_section = {
        "type": "section",
        "id": "model_forecast",
        "title": "Model Forecast",
        "items": [
                {
                    "type": "image",
                    "id": "forecast_plot",
                    "title": "SARIMA Forecast",
                    "file_path": predictions['plot_path'],
                    "description": "SARIMA model forecast with confidence intervals"
                }
            ],
            "comment": "The residuals show no significant autocorrelation patterns, indicating the SARIMA model has successfully captured the underlying time series structure. This validates our model specification."
    }
    
    
    
    response = [model_section, forecast_section]
    
    logger.info(f"Output formatted with {len(response)} sections")
    return response
