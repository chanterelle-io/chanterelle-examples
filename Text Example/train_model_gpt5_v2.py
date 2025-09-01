# from https://www.tensorflow.org/tutorials/keras/text_classification_with_hub (github)
# pip install tensorflow tensorflow-hub tensorflow-datasets tf-keras
import tensorflow as tf
import tensorflow_hub as hub

# from tensorflow import keras
# from tensorflow.keras import layers
import tf_keras as keras

# from tensorflow.keras.datasets import imdb
# Instead of keras.datasets.imdb (which gives int IDs),
# we'll use raw text reviews from TFDS or Hugging Face.
import tensorflow_datasets as tfds

# 1) Load IMDB dataset (as raw text)
(train_data, test_data), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# 2) Build model using TFHub embedding
hub_layer = hub.KerasLayer(
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    input_shape=[],
    dtype=tf.string,
    trainable=True  # freeze embeddings for speed
)

model = keras.Sequential([
    hub_layer,
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# 3) Train
train_data = train_data.shuffle(10000).batch(512).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(512).prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# 4) Evaluate
loss, acc = model.evaluate(test_data)
print(f"Test accuracy: {acc:.4f}")

# 5) Save full model (includes TFHub layer + weights)
model.save("imdb_tfhub_model")
