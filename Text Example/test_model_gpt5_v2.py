import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras

# Later: reload
reloaded = keras.models.load_model("imdb_tfhub_model", custom_objects={"KerasLayer": hub.KerasLayer})

# Test reload
sample_text = tf.constant(["The movie was absolutely fantastic, I loved it!"])
print(reloaded.predict(sample_text))

# Test reload
sample_text = tf.constant(["The movie is rubbish, I hated it."])
print(reloaded.predict(sample_text))