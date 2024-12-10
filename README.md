"# Fashion-MNIST-ANN_CUDA" 

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
