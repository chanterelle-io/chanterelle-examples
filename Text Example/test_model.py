import tensorflow_hub as hub
import tf_keras as keras

model = keras.models.load_model("imdb_tfhub_model", custom_objects={"KerasLayer": hub.KerasLayer})

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