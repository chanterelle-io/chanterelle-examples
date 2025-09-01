# pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1) Load Keras IMDB (already integer-encoded)
vocab_size = 20000
max_len = 250
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# Pad/truncate
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test  = keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=max_len)

# 2) Build model
model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.GlobalMaxPool1D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 3) Train (with a small validation split)
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 4) Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# 5) Save
model.save("imdb_bilstm.keras")
