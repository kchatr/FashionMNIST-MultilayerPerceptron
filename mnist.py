# Libraries for Machine Learning & Data Handling
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper Libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Set-up Logging
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Initialize & Split Dataset 
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_ds, test_ds = dataset['train'], dataset['test']
class_names = metadata.features['label'].names
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

# Function for Normalizing Greyscale Pixel Values
def normalize_data(images):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images

def main():
    # Normalize and Cache Data
    train_ds = train_ds.map(normalize_data)
    test_ds = test_ds.map(normalize_data)
    train_ds = train_ds.cache()
    test_ds = test_ds.cache()

    # Initialize the Model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax())
    ])

    # Compile the Model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy, metrics=['accuracy'])
    
    # Train the Model
    BATCH_SIZE = 32
    train_ds = train_ds.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    test_ds=test_ds.cache().batch(BATCH_SIZE)

    model.fit(train_ds, epochs=5)