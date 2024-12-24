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
```
Loaded 60000 training samples.
Epoch 1, Average Loss: 1.12491, Accuracy: 0.564833, Epoch Time: 512709 ms
Epoch 2, Average Loss: 0.448881, Accuracy: 0.836367, Epoch Time: 509094 ms
Epoch 3, Average Loss: 0.392269, Accuracy: 0.8567, Epoch Time: 509171 ms
Epoch 4, Average Loss: 0.361197, Accuracy: 0.8676, Epoch Time: 510404 ms
Epoch 5, Average Loss: 0.340561, Accuracy: 0.87405, Epoch Time: 512290 ms
Epoch 6, Average Loss: 0.321715, Accuracy: 0.88015, Epoch Time: 510359 ms
Epoch 7, Average Loss: 0.310083, Accuracy: 0.884433, Epoch Time: 511706 ms
Epoch 8, Average Loss: 0.301402, Accuracy: 0.8872, Epoch Time: 508566 ms
Epoch 9, Average Loss: 0.290712, Accuracy: 0.890967, Epoch Time: 512030 ms
Epoch 10, Average Loss: 0.283167, Accuracy: 0.89435, Epoch Time: 514994 ms
```

Explanation:

## **Basic Kernel**

To compile, use this command:
```
nvcc -o basic_kernel BasicKernel.cu
```

To run using the terminal, use:
```
./basic_kernel
```

Result:
```
Loaded 60000 training samples.
Epoch 1, Average Loss: 1.17771, Accuracy: 0.544033, Epoch Time: 16615 ms
Epoch 2, Average Loss: 0.452561, Accuracy: 0.833633, Epoch Time: 16241 ms
Epoch 3, Average Loss: 0.395982, Accuracy: 0.855217, Epoch Time: 16372 ms
Epoch 4, Average Loss: 0.364777, Accuracy: 0.865, Epoch Time: 16296 ms
Epoch 5, Average Loss: 0.343564, Accuracy: 0.873933, Epoch Time: 16287 ms
Epoch 6, Average Loss: 0.325827, Accuracy: 0.8787, Epoch Time: 16393 ms
Epoch 7, Average Loss: 0.311485, Accuracy: 0.88245, Epoch Time: 16423 ms
Epoch 8, Average Loss: 0.299107, Accuracy: 0.88655, Epoch Time: 16268 ms
Epoch 9, Average Loss: 0.290787, Accuracy: 0.891117, Epoch Time: 16409 ms
Epoch 10, Average Loss: 0.279676, Accuracy: 0.894667, Epoch Time: 16318 ms
```

## **Kernel Optimization 1: Input channel reduction using atomics**

To compile, use this command:
```
nvcc -o optimized_kernel OptimizedKernel1.cu
```

To run using the terminal, use:
```
./optimized_kernel
```

Result:
```
Loaded 60000 training samples.
Epoch 1, Average Loss: 1.14309, Accuracy: 0.559633, Epoch Time: 13810 ms
Epoch 2, Average Loss: 0.452262, Accuracy: 0.83555, Epoch Time: 13237 ms
Epoch 3, Average Loss: 0.393042, Accuracy: 0.856067, Epoch Time: 13375 ms
Epoch 4, Average Loss: 0.360136, Accuracy: 0.867167, Epoch Time: 13474 ms
Epoch 5, Average Loss: 0.339493, Accuracy: 0.87435, Epoch Time: 13608 ms
Epoch 6, Average Loss: 0.322844, Accuracy: 0.880333, Epoch Time: 13287 ms
Epoch 7, Average Loss: 0.308822, Accuracy: 0.885117, Epoch Time: 13326 ms
Epoch 8, Average Loss: 0.297592, Accuracy: 0.888, Epoch Time: 13471 ms
Epoch 9, Average Loss: 0.287386, Accuracy: 0.89245, Epoch Time: 13454 ms
Epoch 10, Average Loss: 0.28136, Accuracy: 0.893233, Epoch Time: 13406 ms
```

## **Kernel Optimization 2: Using Streams to overlap computation with data transfer**

To compile, use this command:

```
nvcc -o stream_optimized StreamOptimized.cu
```

To run using the terminal, use:
```
./stream_optimized
```

Result:
```
Loaded 60000 training samples.
Epoch 1, Average Loss: 1.1485, Accuracy: 0.555783, Epoch Time: 16064 ms
Epoch 2, Average Loss: 0.453167, Accuracy: 0.833933, Epoch Time: 15852 ms
Epoch 3, Average Loss: 0.396645, Accuracy: 0.854833, Epoch Time: 16089 ms
Epoch 4, Average Loss: 0.365575, Accuracy: 0.865783, Epoch Time: 15911 ms
Epoch 5, Average Loss: 0.343712, Accuracy: 0.873267, Epoch Time: 15931 ms
Epoch 6, Average Loss: 0.326801, Accuracy: 0.8783, Epoch Time: 16003 ms
Epoch 7, Average Loss: 0.312534, Accuracy: 0.883133, Epoch Time: 16010 ms
Epoch 8, Average Loss: 0.300962, Accuracy: 0.887083, Epoch Time: 15894 ms
Epoch 9, Average Loss: 0.291087, Accuracy: 0.89075, Epoch Time: 15967 ms
Epoch 10, Average Loss: 0.28473, Accuracy: 0.892517, Epoch Time: 16121 ms
```