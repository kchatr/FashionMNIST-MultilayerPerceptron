# FashionMNIST-MultilayerPerceptron

## Preface
Fashion MNIST is a dataset that is modeled after the MNIST dataset, which comprises of 70,000 28x28 pixel images of handwritten digits. Fashion MNIST consists of 70,000 grayscale 28x28 images of various clothing articles, taken from Zalando. There are 10 possible labels for each article. 

The objective of this model is to be able to accurately classify an unknown article of clothing in the same format. 

## Algorithm
This model uses a vanilla multilayer perceptron which uses Sparse Categorical Cross Entropy to evaluate the loss and the Adam optimizer for gradient descent.

The input layer is an array comprising of the normalized grayscale pixel values of the original image which have been flattened. 
The single dense layer has 128 neurons and uses the ReLU activation function. 
The output layer uses the Softmax activation function in order to output a probability distribution representative of the model's confidence that a label is accurate. 

The model is trained over 5 epochs with a batch size of 32.

## Results
The final model has an average testing loss of 0.336 and an accuracy of 0.882. 
Due to the relatively simple architecture of the network and straightforward training process, this is quite a good result.
