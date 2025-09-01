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
    
    # # Create a wrapper class that outputs both species and confidence
    # class IrisClassificationModel:
    #     def __init__(self, model, target_names):
    #         self.model = model
    #         self.target_names = target_names
            
    #     def predict(self, sepal_length, sepal_width, petal_length, petal_width):
    #         """
    #         Predict iris species and confidence
            
    #         Args:
    #             sepal_length (float): Length of sepal in cm
    #             sepal_width (float): Width of sepal in cm
    #             petal_length (float): Length of petal in cm
    #             petal_width (float): Width of petal in cm
                
    #         Returns:
    #             dict: Contains 'species' and 'confidence' keys
    #         """
    #         # Prepare input as array
    #         X_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
    #         # Get prediction and probabilities
    #         prediction = self.model.predict(X_input)[0]
    #         probabilities = self.model.predict_proba(X_input)[0]
            
    #         # Get species name and confidence
    #         species = self.target_names[prediction]
    #         confidence = float(np.max(probabilities))
            
    #         return {
    #             'species': species,
    #             'confidence': confidence
    #         }
        
    #     def predict_batch(self, features_array):
    #         """
    #         Predict for multiple samples
            
    #         Args:
    #             features_array (array-like): Array of shape (n_samples, 4)
                
    #         Returns:
    #             list: List of dictionaries with 'species' and 'confidence'
    #         """
    #         predictions = self.model.predict(features_array)
    #         probabilities = self.model.predict_proba(features_array)
            
    #         results = []
    #         for pred, proba in zip(predictions, probabilities):
    #             species = self.target_names[pred]
    #             confidence = float(np.max(proba))
    #             results.append({
    #                 'species': species,
    #                 'confidence': confidence
    #             })
            
    #         return results
    
    # # Create the wrapped model
    # iris_model = IrisClassificationModel(model, target_names)
    
    # # Test the wrapped model
    # print("\n" + "="*50)
    # print("Testing the wrapped model:")
    
    # # Test with a few examples from the test set
    # for i in range(3):
    #     features = X_test[i]
    #     actual_species = target_names[y_test[i]]
        
    #     result = iris_model.predict(
    #         features[0], features[1], features[2], features[3]
    #     )
        
    #     print(f"\nSample {i+1}:")
    #     print(f"  Features: sepal_length={features[0]:.2f}, sepal_width={features[1]:.2f}, "
    #           f"petal_length={features[2]:.2f}, petal_width={features[3]:.2f}")
    #     print(f"  Actual species: {actual_species}")
    #     print(f"  Predicted species: {result['species']}")
    #     print(f"  Confidence: {result['confidence']:.4f}")
    
    iris_model = model  # Use the original model for saving
    
    # Save the model
    model_path = 'iris_classification_model.joblib'
    print(f"\nSaving model to {model_path}...")
    joblib.dump(iris_model, model_path)
    
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    return iris_model, model_path

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