import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import joblib
import os

def create_and_save_sarima_model():
    """Create, fit, and save the SARIMA model for the handler"""
    try:
        # Load airline passenger dataset
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
        
        print("Data loaded successfully")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Fit SARIMA model
        print("Fitting SARIMA(1,1,1)(1,1,1,12) model...")
        sarima_model = sm.tsa.statespace.SARIMAX(
            df["Passengers"],
            order=(1,1,1),           # ARIMA part
            seasonal_order=(1,1,1,12) # seasonal part
        ).fit()
        
        print("Model fitted successfully")
        print(f"AIC: {sarima_model.aic:.2f}")
        print(f"BIC: {sarima_model.bic:.2f}")
        
        # Save model data
        model_data = {
            'model': sarima_model,
            'original_data': df,
            'last_date': df.index[-1]
        }
        
        model_path = os.path.join(os.path.dirname(__file__), 'sarima_model.joblib')
        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")
        
        # Test forecast
        forecast = sarima_model.get_forecast(steps=12)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        print(f"Test forecast (12 months):")
        print(f"First forecast: {forecast_mean.iloc[0]:.1f}")
        print(f"Last forecast: {forecast_mean.iloc[-1]:.1f}")
        
        return model_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    create_and_save_sarima_model()
