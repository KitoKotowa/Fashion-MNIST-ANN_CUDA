# Neural network with CUDA using Fashion-MNIST Dataset

## **Architecture**
- This is a traditional Artificial Neural Network (ANN) which consist of: 1 input layer, 2 hidden layers, 1 output layer.

```
Layer (type)                Output Shape              Param #    
================================================================= 
flatten (Flatten)           (None, 784)               0          
_________________________________________________________________ 
dense (Dense)               (None, 128)               100480     
_________________________________________________________________ 
dense_1 (Dense)             (None, 128)               16512      
_________________________________________________________________ 
dense_2 (Dense)             (None, 10)                1290 
```

- The activation function on 2 hidden layers is ReLU function 
- The activation function on output layer is softmax function 

## **Dataset**
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 

## **Host Code**
First you need to download the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page#Download) library, for matrix operations

Setup Eigen:
- No installation is required for Eigen as it is a header-only library.
- Simply include the path to the Eigen directory in your project's include path.

To compile the host code, use this command:
```
g++ fashion_mnist_host.cpp -I"Path\Of\Eigen\eigen-3.4.0" -o fashion_mnist_host
```

To run using the terminal, use:
```
./fashion_mnist_host
```

Result:

Explanation:

## **Basic Kernel**

Result:
```

```
Explanation:

## **Kernel Optimization 1: Input channel reduction using atomics**

Result:
```

```
Explanation: 

## **Kernel Optimization 2: Using Streams to overlap computation with data transfer**

Result:
```

```
Explanation: