# IMDB Movie Review Sentiment Classifier - Basic Tutorial
# Based on: https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
# Required packages: pip install tensorflow tensorflow-hub tensorflow-datasets tf-keras

import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import tensorflow_datasets as tfds
import json
import os

# 1) Load IMDB dataset (as raw text)
print("Loading IMDB dataset...")
(train_data, test_data), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print(f"Training examples: {ds_info.splits['train'].num_examples}")
print(f"Test examples: {ds_info.splits['test'].num_examples}")

# Show a sample review
for review, label in train_data.take(1):
    print(f"\nSample review: {review.numpy().decode('utf-8')[:200]}...")
    print(f"Label: {label.numpy()} ({'Positive' if label.numpy() == 1 else 'Negative'})")

# 2) Build model using TFHub embedding
print("\nBuilding model with TensorFlow Hub embeddings...")
hub_layer = hub.KerasLayer(
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    input_shape=[],
    dtype=tf.string,
    trainable=True  # Allow fine-tuning of embeddings
)

model = keras.Sequential([
    hub_layer,                                    # Pre-trained word embeddings
    keras.layers.Dense(16, activation="relu"),    # Hidden layer
    keras.layers.Dropout(0.2),                   # Prevent overfitting
    keras.layers.Dense(1, activation="sigmoid")   # Output: 0=negative, 1=positive
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("\nModel architecture:")
model.summary()

# 3) Prepare data and train
print("\nPreparing data...")
train_data = train_data.shuffle(10000).batch(512).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(512).prefetch(tf.data.AUTOTUNE)

print("Training model...")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    verbose=1
)

# 4) Evaluate
print("\nEvaluating model...")
loss, acc = model.evaluate(test_data, verbose=1)
print(f"\nFinal Results:")
print(f"Test accuracy: {acc:.4f}")
print(f"Test loss: {loss:.4f}")

# 5) Save model and info
print("\nSaving model...")
model.save("imdb_tfhub_model")

# Save model info
model_info = {
    "model_type": "IMDB Sentiment Classifier (TensorFlow Hub)",
    "architecture": "Sequential with TFHub embeddings",
    "embedding_model": "nnlm-en-dim50",
    "input": "raw_text",
    "output": "sentiment (0=negative, 1=positive)",
    "test_accuracy": float(acc),
    "test_loss": float(loss),
    "epochs": 10
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"Model saved to: imdb_tfhub_model/")
print(f"Model size: {sum(os.path.getsize(os.path.join('imdb_tfhub_model', f)) for f in os.listdir('imdb_tfhub_model') if os.path.isfile(os.path.join('imdb_tfhub_model', f))) / 1024:.2f} KB")
print(f"Metadata saved to: model_info.json")

# 6) Test with example predictions
print("\n" + "="*60)
print("Example Predictions:")
print("-" * 60)

test_reviews = [
    "This movie was absolutely fantastic! Great acting and amazing plot.",
    "Terrible movie. Waste of time. Poor acting and boring storyline.",
    "The movie was okay. Not great but not terrible either."
]

for review in test_reviews:
    prediction = model.predict([review])[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\nReview: '{review}'")
    print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
