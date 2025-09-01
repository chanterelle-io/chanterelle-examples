import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
import re

def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_imdb_data():
    """
    Load IMDB dataset using TensorFlow datasets
    """
    print("Loading IMDB movie reviews dataset...")
    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=10000,  # Only consider the top 10,000 words
        skip_top=0,       # Don't skip any words
        maxlen=None,      # Don't limit sequence length yet
        start_char=1,     # Start character
        oov_char=2,       # Out-of-vocabulary character
        index_from=3      # Index actual words from 3
    )
    
    # Get word index mapping
    word_index = tf.keras.datasets.imdb.get_word_index()
    
    # Create reverse word index (from integer back to word)
    reverse_word_index = {value: key for key, value in word_index.items()}
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Vocabulary size: {len(word_index)}")
    
    return (x_train, y_train), (x_test, y_test), word_index, reverse_word_index

def decode_review(encoded_review, reverse_word_index):
    """
    Convert encoded review back to text
    """
    return ' '.join([reverse_word_index.get(i, '<UNK>') for i in encoded_review])

def train_imdb_classifier():
    """
    Train an IMDB movie review sentiment classifier
    """
    print("IMDB Movie Review Sentiment Classification")
    print("=" * 50)
    
    # Load data
    (x_train, y_train), (x_test, y_test), word_index, reverse_word_index = load_imdb_data()
    
    # Show some sample data
    print(f"\nSample review (encoded): {x_train[0][:20]}...")
    print(f"Sample review (decoded): {decode_review(x_train[0], reverse_word_index)[:100]}...")
    print(f"Sample label: {y_train[0]} ({'Positive' if y_train[0] == 1 else 'Negative'})")
    
    # Pad sequences to have the same length
    maxlen = 500  # Maximum review length
    print(f"\nPadding sequences to length {maxlen}...")
    
    x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=maxlen, padding='post', truncating='post'
    )
    x_test_padded = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=maxlen, padding='post', truncating='post'
    )
    
    # Split training data into train and validation
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train_padded, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(x_train_final)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test_padded)}")
    
    # Build the model
    print("\nBuilding neural network model...")
    
    model = models.Sequential([
        # Embedding layer
        layers.Embedding(input_dim=10000, output_dim=128, input_length=maxlen),
        
        # Add dropout for regularization
        layers.Dropout(0.5),
        
        # Global average pooling
        layers.GlobalAveragePooling1D(),
        
        # Dense hidden layer
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    
    history = model.fit(
        x_train_final, y_train_final,
        epochs=10,
        batch_size=512,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test_padded, y_test, verbose=1)
    
    # Make predictions
    y_pred_proba = model.predict(x_test_padded)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Show some prediction examples
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("-" * 80)
    
    for i in range(5):
        review_text = decode_review(x_test[i], reverse_word_index)
        actual_sentiment = 'Positive' if y_test[i] == 1 else 'Negative'
        predicted_sentiment = 'Positive' if y_pred[i] == 1 else 'Negative'
        confidence = y_pred_proba[i][0] if y_pred[i] == 1 else 1 - y_pred_proba[i][0]
        
        print(f"\nSample {i+1}:")
        print(f"Review: {review_text[:200]}...")
        print(f"Actual: {actual_sentiment}")
        print(f"Predicted: {predicted_sentiment} (confidence: {confidence:.3f})")
        print(f"Correct: {'✓' if actual_sentiment == predicted_sentiment else '✗'}")
    
    # Save the model
    model_path = 'imdb_sentiment_model.h5'
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    
    # Save tokenizer information
    tokenizer_data = {
        'word_index': word_index,
        'reverse_word_index': reverse_word_index,
        'maxlen': maxlen,
        'vocab_size': 10000
    }
    
    tokenizer_path = 'imdb_tokenizer.joblib'
    joblib.dump(tokenizer_data, tokenizer_path)
    
    # Save model metadata
    model_info = {
        'model_type': 'IMDB Sentiment Classifier',
        'architecture': 'Sequential Neural Network with Embedding',
        'input_features': ['movie_review_text'],
        'output': 'sentiment (0=negative, 1=positive)',
        'vocabulary_size': 10000,
        'max_sequence_length': maxlen,
        'metrics': {
            'test_accuracy': float(accuracy),
            'test_loss': float(test_loss)
        },
        'training_info': {
            'epochs': 10,
            'batch_size': 512,
            'optimizer': 'adam',
            'loss_function': 'binary_crossentropy'
        }
    }
    
    with open('model_meta.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Model metadata saved to: model_meta.json")
    
    return model, tokenizer_data, model_path

def predict_sentiment(model, tokenizer_data, review_text):
    """
    Predict sentiment for a new review
    
    Args:
        model: Trained model
        tokenizer_data: Tokenizer information
        review_text: Raw review text
        
    Returns:
        tuple: (sentiment, confidence)
    """
    # Preprocess the text
    cleaned_text = preprocess_text(review_text)
    
    # Convert text to sequence
    word_index = tokenizer_data['word_index']
    maxlen = tokenizer_data['maxlen']
    
    # Convert words to integers
    sequence = []
    for word in cleaned_text.split():
        if word in word_index:
            if word_index[word] < 10000:  # Only use words in vocabulary
                sequence.append(word_index[word])
            else:
                sequence.append(2)  # OOV token
        else:
            sequence.append(2)  # OOV token
    
    # Pad sequence
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence], maxlen=maxlen, padding='post', truncating='post'
    )
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence

if __name__ == "__main__":
    try:
        model, tokenizer_data, model_path = train_imdb_classifier()
        
        print("\n" + "="*80)
        print("Training completed successfully!")
        print(f"Model saved as: {model_path}")
        print("\nTo use the model for predictions:")
        print("  import tensorflow as tf")
        print("  import joblib")
        print(f"  model = tf.keras.models.load_model('{model_path}')")
        print("  tokenizer_data = joblib.load('imdb_tokenizer.joblib')")
        print("  # Example prediction:")
        print("  sentiment, confidence = predict_sentiment(model, tokenizer_data, 'This movie was amazing!')")
        
        # Example predictions
        print("\n" + "="*80)
        print("Example Predictions:")
        
        test_reviews = [
            "This movie was absolutely fantastic! Great acting and amazing plot.",
            "Terrible movie. Waste of time. Poor acting and boring storyline.",
            "The movie was okay. Not great but not terrible either.",
            "Best film I've ever seen! Highly recommend it to everyone.",
            "Completely disappointing. Expected much more from this director."
        ]
        
        for review in test_reviews:
            sentiment, confidence = predict_sentiment(model, tokenizer_data, review)
            print(f"\nReview: '{review}'")
            print(f"Predicted: {sentiment} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise