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

# Function for Normalizing Greyscale Pixel Values
def normalize_data(images):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images

def plot_image(index, pred_arr, labels, images):
    predictions_array, label, img = pred_arr[index], labels[index], images[index] 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                            100*np.max(predictions_array),
                            class_names[label]),
                            color=color)

def plot_val_array(index, pred_arr, labels):
    predictions_array, label = pred_arr[index], labels[index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[label].set_color('blue')

def main():
    # Initialize & Split Dataset 
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_ds, test_ds = dataset['train'], dataset['test']
    class_names = metadata.features['label'].names
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples

    # Normalize and Cache Data
    train_ds = train_ds.map(normalize_data)
    test_ds = test_ds.map(normalize_data)
    train_ds = train_ds.cache()
    test_ds = test_ds.cache()

    # Explore Data
    # Print the breakdown of training data to testing data.
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    print("Number of training examples: {}".format(num_train_examples))
    print("Number of test examples:     {}".format(num_test_examples))

    # Display the first 25 training examples
    plt.figure(figsize=(15,15))
    for i, (image, label) in enumerate(train_ds.take(25)):
        image = image.numpy().reshape((28,28))
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
    plt.show()

    # Plot the image - voila a piece of fashion clothing
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

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

    # Evaluate Model Accuracy
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Testing Loss and Accuracy: {test_loss}; {test_accuracy}")

    # Generate Predictions on Test Set
    for test_images, test_labels in test_ds.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)

    # Plot Model Predictions
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_val_array(i, predictions, test_labels)

