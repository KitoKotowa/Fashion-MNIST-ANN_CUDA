#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

class NeuralNetwork {
private:
    const int input_size = 784;
    const int hidden1_size = 128;
    const int hidden2_size = 128; 
    const int output_size = 10;
    const double learning_rate = 0.01;

    MatrixXd W1, W2, W3;
    VectorXd b1, b2, b3;

    // Activation function (ReLU)
    MatrixXd relu(const MatrixXd& x) {
        return x.array().max(0);
    }

    MatrixXd relu_derivative(const MatrixXd& x) {
        return (x.array() > 0).cast<double>();
    }

    // Softmax activation
    MatrixXd softmax(const MatrixXd& x) {
        MatrixXd exp_x = x.array().exp();
        return exp_x.array().colwise() / exp_x.colwise().sum().array();
    }

public:
    NeuralNetwork() {
        // Initialize random number generator
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, sqrt(2.0/input_size));

        // Initialize weights and biases
        W1 = MatrixXd::Zero(hidden1_size, input_size);
        W2 = MatrixXd::Zero(hidden2_size, hidden1_size);
        W3 = MatrixXd::Zero(output_size, hidden2_size);
        
        b1 = VectorXd::Zero(hidden1_size);
        b2 = VectorXd::Zero(hidden2_size);
        b3 = VectorXd::Zero(output_size);

        // Random initialization
        for(int i = 0; i < W1.rows(); i++)
            for(int j = 0; j < W1.cols(); j++)
                W1(i,j) = dist(gen);

        for(int i = 0; i < W2.rows(); i++)
            for(int j = 0; j < W2.cols(); j++)
                W2(i,j) = dist(gen);

        for(int i = 0; i < W3.rows(); i++)
            for(int j = 0; j < W3.cols(); j++)
                W3(i,j) = dist(gen);
    }

    MatrixXd forward(const MatrixXd& X) {
        // Forward propagation
        MatrixXd z1 = W1 * X + b1 * MatrixXd::Ones(1, X.cols());
        MatrixXd a1 = relu(z1);
        
        MatrixXd z2 = W2 * a1 + b2 * MatrixXd::Ones(1, X.cols());
        MatrixXd a2 = relu(z2);
        
        MatrixXd z3 = W3 * a2 + b3 * MatrixXd::Ones(1, X.cols());
        return softmax(z3);
    }

    void train(const MatrixXd& X, const MatrixXd& y, int epochs, int batch_size) {
        int n_samples = X.cols();
        int n_batches = n_samples / batch_size;

        for(int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0;

            for(int batch = 0; batch < n_batches; batch++) {
                int start_idx = batch * batch_size;
                MatrixXd X_batch = X.block(0, start_idx, X.rows(), batch_size);
                MatrixXd y_batch = y.block(0, start_idx, y.rows(), batch_size);

                // Forward pass
                MatrixXd z1 = W1 * X_batch + b1 * MatrixXd::Ones(1, batch_size);
                MatrixXd a1 = relu(z1);
                
                MatrixXd z2 = W2 * a1 + b2 * MatrixXd::Ones(1, batch_size);
                MatrixXd a2 = relu(z2);
                
                MatrixXd z3 = W3 * a2 + b3 * MatrixXd::Ones(1, batch_size);
                MatrixXd yhat = softmax(z3);

                // Backward pass
                MatrixXd delta3 = yhat - y_batch;
                MatrixXd delta2 = (W3.transpose() * delta3).array() * relu_derivative(z2).array();
                MatrixXd delta1 = (W2.transpose() * delta2).array() * relu_derivative(z1).array();

                // Update weights and biases
                W3 -= learning_rate * (delta3 * a2.transpose()) / batch_size;
                W2 -= learning_rate * (delta2 * a1.transpose()) / batch_size;
                W1 -= learning_rate * (delta1 * X_batch.transpose()) / batch_size;
                
                b3 -= learning_rate * delta3.rowwise().mean();
                b2 -= learning_rate * delta2.rowwise().mean();
                b1 -= learning_rate * delta1.rowwise().mean();

                // Calculate loss
                total_loss -= (y_batch.array() * yhat.array().log()).sum() / batch_size;
            }

            if(epoch % 5 == 0) {
                cout << "Epoch " << epoch << ", Loss: " << total_loss/n_batches << endl;
            }
        }
    }
};

__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void relu_derivative_kernel(float* input, float* output, int size) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

__global__ void softmax_kernel(float* input, float* output, int size) {
    __shared__ float max_val;
    __shared__ float sum;

    if (threadIdx.x == 0) {
        max_val = input[0];
        for  (int i = 1; i < size; ++size) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
    }

    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float exp_x = idx < size ? exp(input[idx] - max_val) : 0.0f;
    
    __shared__ float shared_mem[BLOCK_SIZE];
    shared_mem[threadIdx.x] = exp_x;

    for (int stride = blockDim.x / 2; stride > 0; stride >> 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
    }
    
    if (threadIdx.x == 0) {
        sum = shared_mem[0];
    }

    __syncthreads();

    if (idx < size) {
        output[idx] = exp(input[idx] - max_val) / sum;
    }
}

bool loadFashionMNIST(vector<vector<double>>& images,
                      vector<int>& labels,
                      const string& image_file,
                      const string& label_file) {
    ifstream file_images(image_file, ios::binary); 
    ifstream file_labels(label_file, ios::binary);
    
    if(!file_images.is_open() || !file_labels.is_open()) {
        cout << "Error opening dataset files" << endl;
        return false;
    }
    
    // Read image file header
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file_images.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file_images.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file_images.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file_images.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    
    // Read label file header
    file_labels.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file_labels.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    
    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
    
    // Read images and labels
    images.resize(num_images, vector<double>(num_rows * num_cols));
    labels.resize(num_images);
    
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel = 0;
            file_images.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            if (file_images.fail()) {
                std::cerr << "Error reading image data." << std::endl;
                return false;
            }
            images[i][j] = pixel / 255.0; // Normalize to [0, 1]
        }

        unsigned char label = 0;
        file_labels.read(reinterpret_cast<char*>(&label), sizeof(label));
        if (file_labels.fail()) {
            std::cerr << "Error reading label data." << std::endl;
            return false;
        }
        labels[i] = static_cast<int>(label);
    }
    
    return true;
}

void printTrainingData(const std::vector<std::vector<double>>& images,
                       const std::vector<int>& labels, int count) {
    for (int i = 0; i < count && i < images.size(); ++i) {
        cout.precision(2);
        std::cout << "Sample " << i + 1 << ":\n";
        std::cout << "Label: " << labels[i] << "\n";
        std::cout << "Image (Flattened):\n";
        for (int j = 0; j < images[i].size(); ++j) {
            std::cout << images[i][j] << " ";
            if ((j + 1) % 28 == 0) std::cout << "\n"; // Format into 28 pixels per row
        }
        std::cout << "\n" << std::string(40, '-') << "\n";
    }
}

int main() {
    vector<vector<double>> images;
    vector<int> labels;
    
    string image_file = "train-images-idx3-ubyte";
    string label_file = "train-labels-idx1-ubyte";
    
    if (!loadFashionMNIST(images, labels, image_file, label_file)) {
        cerr << "Failed to load Fashion MNIST dataset" << endl;
        return 1;
    }

    cout << "Loaded " << images.size() << " training samples.\n";
    cout << "Printing the first 5 samples:\n"; 
    printTrainingData(images, labels, 5);

    // Convert vector<vector<double>> to Eigen::MatrixXd
    MatrixXd training_data(images.size(), images[0].size());
    for (size_t i = 0; i < images.size(); i++) {
        for (size_t j = 0; j < images[i].size(); j++) {
            training_data(i, j) = images[i][j];
        }
    }

    // Convert labels to one-hot encoded matrix
    MatrixXd training_labels = MatrixXd::Zero(labels.size(), 10);
    for (size_t i = 0; i < labels.size(); i++) {
        training_labels(i, labels[i]) = 1.0;
    }

    // Create and train neural network
    NeuralNetwork nn;
    cout << "Training neural network...\n";
    nn.train(training_data, training_labels);
    cout << "Training complete!\n";
    
    return 0;
}