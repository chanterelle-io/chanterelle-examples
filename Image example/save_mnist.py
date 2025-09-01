import tensorflow as tf
import os
from PIL import Image

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def save_images(images, labels, folder):
    os.makedirs(folder, exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels)):
        label_folder = os.path.join(folder, str(label))
        os.makedirs(label_folder, exist_ok=True)
        Image.fromarray(img).save(os.path.join(label_folder, f"{i}.png"))

save_images(x_train, y_train, "mnist_train")
save_images(x_test, y_test, "mnist_test")
