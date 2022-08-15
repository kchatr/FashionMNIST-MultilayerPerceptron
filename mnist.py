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

# Function for Normalizing Greyscale Pixel Values
def normalize_data(images):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images



