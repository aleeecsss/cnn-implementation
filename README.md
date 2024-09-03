# cnn-implementation

This is a humbling attempt of implementing Convolutional Neural Networks (or CNNs) in C++ language, without other external libraries. I tried to accurately solve the MNIST classifying-digit problem.

The following project contains:

- Simple, dense layers that fully connect a layer with another one.
- Convolutional layers, used in image processing, that check for specific shapes in small continuous submatrices of the image.
- Pooling layers (Max-Pool is used), also used in image processing, that compress the image, each new neuron in this layer representing the highest activation from a small continuous submatrix.
- Sigmoid and ReLU activation functions.
- Softmax layer that transforms the final layer in a neural network in a probability distribution (usually used in decision problems).
- Log-likelihood loss function.
- Auto-differentation (Feedforward and Backpropagation algorithms for each type of layer and activation function).
- Parsing function for reading the training / testing data.
- A momentum-based stochastic gradient descent optimizer.

# How to use?

Step-by-step tutorial:

Step 0: Implement the parseData function in main.cpp for your special training / testing data. The parseData has a special parameter "lmt" that limits the amount of tests read by: min {lmt, trainingDataSize / testingDataSize}.
Step 1: Create your custom neural network.

- Inserting a dense layer:
  ```cpp
      net.push_back(std::make_unique<denseLayer>(DIMENSION_1, DIMENSION_2));
  ```

  Creates a dense layer that completely connects DIMENSION_1 neurons in a layer with other DIMENSION_2 neurons in another layer.

- Inserting a convolutional layer:
  ```cpp
      net.push_back(std::make_unique<convolutionLayer>(HEIGHT, WIDTH, K, MAPS));
  ```

  Creates a convolutional layer that connects a layer of neurons that represents a HEIGHT x WIDTH image and constructs the following layer, that represents the activation for each K x K continuous submatrix, along MAPS feature maps (each feature map detects a different kind of shape). There are (HEIGHT - K + 1) x (WIDTH - K + 1) neurons in the following layer.

- Inserting a pooling layer:
  ```cpp
      net.push_back(std::make_unique<poolingLayer>(HEIGHT, WIDTH, K, MAPS));
  ```

  Creates a pooling layer that connects a layer of neurons that represents a HEIGHT x WIDTH image and constructs the following layer, that represents a (HEIGHT / K) x (WIDTH / K) compressed image (Attention: HEIGHT and WIDTH must be divisible by K) along MAPS feature maps. Pooling layers are usually used after a convolutional Layer. (Attention: the MAPS value must be constant between corresponding convolutional and pooling layers).

- Inserting a Sigmoid activation layer:
  ```cpp
      net.push_back(std::make_unique<Sigmoid>(DIMENSION));
  ```

  Creates an activation layer for DIMENSION neurons.

- Inserting a ReLU activation layer:
  ```cpp
      net.push_back(std::make_unique<ReLU>(DIMENSION));
  ```

  Creates an activation layer for DIMENSION neurons.

- Inserting a Softmax layer:
  ```cpp
      net.push_back(std::make_unique<Softmax>(DIMENSION));
  ```

  Creates a layer that constructs a probability distribution of the activation of the last layer of neurons (Attention: MUST be used at the end of a neural network because of the loss function; If you want to get rid of this restriction, change the loss function).

Step 2: Tune your hyperparameters.

```cpp
  void stochasticGradientDescent(int mode, int cntEpochs, int subsetSize, double eta, double micro, double lambda) {}
```

- mode = 0: Creates a validation data array, equal in length with the testing data array. Used for choosing the hyperparameters.
- mode = 1: Training and testing the AI.
- cntEpochs: The number of epochs in which the model is trained.
- subsetSize: The dimension of the minibatches used in training.
- eta: Learning rate of the gradient descent.
- micro: Friction quotient for momentum.
- lambda: The hyperparameter used in L2-regularization.

# Information regarding performance

This code is NOT CUDA-optimized. Therefore, it runs on your CPU. Expect lower efficiency.

# Accuracy

The convolutional neural network (with all hyperparameters) used in main.cpp achieves >95% accuracy only after 5 epochs of training.
It should achieve ~99% accuracy after 30+ epochs of training, but I was too lazy to try it out. 😹

# TO-DO

- Add dropout
- Test L2-regularization
