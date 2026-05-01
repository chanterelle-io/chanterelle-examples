import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_iris_model():
    """
    Train an iris classification model and save it as joblib
    """
    print("Loading iris dataset...")
    
    # Load the iris dataset
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)
    
    # Feature names for reference
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_names = ['setosa', 'versicolor', 'virginica']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Feature importance
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"{name}: {importance:.4f}")
            
    # Save the model
    model_path = 'iris_classification_model.joblib'
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    return model, model_path

if __name__ == "__main__":
    print("Iris Classification Model Training")
    print("=" * 40)
    
    try:
        model, model_path = train_iris_model()
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Model saved as: {model_path}")
        print("\nTo use the model:")
        print("  import joblib")
        print(f"  model = joblib.load('{model_path}')")
        print("  result = model.predict(5.1, 3.5, 1.4, 0.2)")
        print("  print(result)  # {'species': 'setosa', 'confidence': 0.95}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise