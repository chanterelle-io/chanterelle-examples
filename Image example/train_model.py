import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt

def train_mnist_model():
    """
    Train an MNIST digit classification model and save it
    """
    print("Loading MNIST dataset...")
    
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Normalize pixel values to 0-1 range
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape data to add channel dimension (for CNN)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical one-hot encoding
    y_train_categorical = keras.utils.to_categorical(y_train, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)
    
    print(f"Preprocessed training data shape: {X_train.shape}")
    print(f"Preprocessed test data shape: {X_test.shape}")
    
    # Create CNN model
    print("\nBuilding CNN model...")
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train_categorical,
        batch_size=128,
        epochs=10,
        validation_data=(X_test, y_test_categorical),
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    target_names = [str(i) for i in range(10)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Test the model with a few examples
    print("\n" + "="*50)
    print("Testing the model:")
    
    for i in range(3):
        test_image = X_test[i:i+1]  # Keep batch dimension
        actual_digit = y_test[i]
        
        # Make prediction
        probabilities = model.predict(test_image, verbose=0)[0]
        predicted_digit = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        print(f"\nSample {i+1}:")
        print(f"  Actual digit: {actual_digit}")
        print(f"  Predicted digit: {predicted_digit}")
        print(f"  Confidence: {confidence:.4f}")
    
    # Save the model
    model_path = 'mnist_classification_model.h5'
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, model_path

if __name__ == "__main__":
    print("MNIST Digit Classification Model Training")
    print("=" * 45)
    
    try:
        model, model_path = train_mnist_model()
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Keras model saved as: {model_path}")
        print("\nTo use the model:")
        print("  import tensorflow as tf")
        print(f"  model = tf.keras.models.load_model('{model_path}')")
        print("  # Preprocessing and prediction will be handled in handler_io.py")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise