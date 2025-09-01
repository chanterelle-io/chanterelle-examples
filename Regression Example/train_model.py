import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def train_california_housing_model():
    """
    Train a California housing price prediction model and save it as joblib
    """
    print("Loading California housing dataset...")
    
    # Load the California housing dataset
    housing = fetch_california_housing()
    X = housing.data  # Features
    y = housing.target  # Target: median house value in hundreds of thousands of dollars
    
    # Feature names for reference
    feature_names = [
        'median_income',        # Median income in block group
        'house_age',           # Median house age in block group
        'avg_rooms',           # Average number of rooms per household
        'avg_bedrooms',        # Average number of bedrooms per household
        'population',          # Block group population
        'avg_occupancy',       # Average number of household members
        'latitude',            # Block group latitude
        'longitude'            # Block group longitude
    ]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target: Median house value (in hundreds of thousands of dollars)")
    print(f"Target range: ${y.min():.2f}k - ${y.max():.2f}k")
    print(f"Target mean: ${y.mean():.2f}k")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\nTraining Random Forest Regression model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}k")
    print(f"MAE: ${mae:.2f}k")
    print(f"MSE: {mse:.4f}")
    
    # # Feature importance
    # print("\nFeature Importance:")
    # feature_importance = list(zip(feature_names, model.feature_importances_))
    # feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # for name, importance in feature_importance:
    #     print(f"{name}: {importance:.4f}")
    
    # # Show some prediction examples
    # print("\n" + "="*60)
    # print("Sample Predictions:")
    # print("-" * 60)
    
    # for i in range(5):
    #     features = X_test[i]
    #     actual_price = y_test[i]
    #     predicted_price = y_pred[i]
    #     error = abs(actual_price - predicted_price)
        
    #     print(f"\nSample {i+1}:")
    #     print(f"  Median Income: ${features[0]:.2f} (tens of thousands)")
    #     print(f"  House Age: {features[1]:.1f} years")
    #     print(f"  Avg Rooms: {features[2]:.1f}")
    #     print(f"  Latitude: {features[6]:.2f}, Longitude: {features[7]:.2f}")
    #     print(f"  Actual Price: ${actual_price:.2f}k")
    #     print(f"  Predicted Price: ${predicted_price:.2f}k")
    #     print(f"  Error: ${error:.2f}k")
    
    # Save the model
    model_path = 'california_housing_model.joblib'
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    # Create and save feature names for reference
    feature_info = {
        'feature_names': feature_names,
        'feature_descriptions': {
            'median_income': 'Median income in block group (tens of thousands of dollars)',
            'house_age': 'Median house age in block group (years)',
            'avg_rooms': 'Average number of rooms per household',
            'avg_bedrooms': 'Average number of bedrooms per household', 
            'population': 'Block group population',
            'avg_occupancy': 'Average number of household members',
            'latitude': 'Block group latitude',
            'longitude': 'Block group longitude'
        },
        'target_description': 'Median house value in hundreds of thousands of dollars',
        'model_metrics': {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse)
        }
    }
    
    import json
    with open('model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    return model, model_path

def predict_house_price(model, median_income, house_age, avg_rooms, avg_bedrooms, 
                       population, avg_occupancy, latitude, longitude):
    """
    Predict house price using the trained model
    
    Args:
        model: Trained model
        median_income (float): Median income in block group (tens of thousands)
        house_age (float): Median house age in block group
        avg_rooms (float): Average number of rooms per household
        avg_bedrooms (float): Average number of bedrooms per household
        population (float): Block group population
        avg_occupancy (float): Average number of household members
        latitude (float): Block group latitude
        longitude (float): Block group longitude
        
    Returns:
        float: Predicted house price in hundreds of thousands of dollars
    """
    # Prepare input as array
    X_input = np.array([[median_income, house_age, avg_rooms, avg_bedrooms,
                        population, avg_occupancy, latitude, longitude]])
    
    # Get prediction
    prediction = model.predict(X_input)[0]
    
    return prediction

if __name__ == "__main__":
    print("California Housing Price Prediction Model Training")
    print("=" * 50)
    
    try:
        model, model_path = train_california_housing_model()
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Model saved as: {model_path}")
        print("\nTo use the model:")
        print("  import joblib")
        print(f"  model = joblib.load('{model_path}')")
        print("  # Example prediction:")
        print("  price = model.predict([[8.3252, 41.0, 6.98, 1.02, 322.0, 2.56, 37.88, -122.23]])")
        print("  print(f'Predicted price: ${price[0]:.2f}k')")
        
        # Example prediction
        print("\n" + "="*60)
        print("Example Prediction:")
        example_price = predict_house_price(
            model, 
            median_income=8.33,    # $83,300 median income
            house_age=41.0,        # 41 years old
            avg_rooms=6.98,        # ~7 rooms per household
            avg_bedrooms=1.02,     # ~1 bedroom per household
            population=322.0,      # 322 people in block group
            avg_occupancy=2.56,    # ~2.6 people per household
            latitude=37.88,        # San Francisco Bay Area
            longitude=-122.23
        )
        print(f"Predicted house price: ${example_price:.2f}k (${example_price*100:.0f})")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
