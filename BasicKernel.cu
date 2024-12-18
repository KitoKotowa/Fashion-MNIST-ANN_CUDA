#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define BLOCK_SIZE 256 
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "; \
        cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << endl; \
        throw runtime_error(cudaGetErrorString(error)); \
    } \
}

class NeuralNetworkCUDA {
private:
    const int input_size = 784;
    const int hidden1_size = 128;
    const int hidden2_size = 128; 
    const int output_size = 10;
    const float learning_rate = 0.01;

    float *d_W1, *d_W2, *d_W3;
    float *d_b1, *d_b2, *d_b3;
    float *d_input, *d_z1, *d_a1, *d_z2, *d_a2, *d_z3, *d_output;
    float *d_target, *d_dZ3, *d_dZ2, *d_dZ1;

    vector<float> h_W1, h_W2, h_W3;
    vector<float> h_b1, h_b2, h_b3;

    void allocate_memory() {
        // Weights and biases
        cudaMalloc(&d_W1, hidden1_size * input_size * sizeof(float));
        cudaMalloc(&d_W2, hidden2_size * hidden1_size * sizeof(float));
        cudaMalloc(&d_W3, output_size * hidden2_size * sizeof(float));
        
        cudaMalloc(&d_b1, hidden1_size * sizeof(float));   
        cudaMalloc(&d_b2, hidden2_size * sizeof(float));
        cudaMalloc(&d_b3, output_size * sizeof(float));
    }

    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> d(0, 0.01);

        // Xavier/Glorot initialization
        h_W1.resize(hidden1_size * input_size);
        h_W2.resize(hidden2_size * hidden1_size);
        h_W3.resize(output_size * hidden2_size);
        
        h_b1.resize(hidden1_size, 0.0f);
        h_b2.resize(hidden2_size, 0.0f);
        h_b3.resize(output_size, 0.0f);

        for (int i = 0; i < h_W1.size(); ++i) {
            h_W1[i] = d(gen) * sqrt(2.0f / (input_size + hidden1_size));
        }

        for (int i = 0; i < h_W2.size(); ++i) {
            h_W2[i] = d(gen) * sqrt(2.0f / (hidden1_size + hidden2_size));
        }

        for (int i = 0; i < h_W3.size(); ++i) {
            h_W3[i] = d(gen) * sqrt(2.0f / (hidden2_size + output_size));
        }

        cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W3, h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    void forward_propagation() {
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // First hidden layer
        // TODO: Implement CUDA kernels for matrix multiplication
        // Placeholder for CUDA gemm (general matrix multiplication)
        
        // ReLU activation
        int blocks1 = (hidden1_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relu_kernel<<<blocks1, BLOCK_SIZE>>>(d_z1, d_a1, hidden1_size);
        
        // Second hidden layer
        // TODO: Similar matrix multiplication kernel
        
        // ReLU activation
        int blocks2 = (hidden2_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relu_kernel<<<blocks2, BLOCK_SIZE>>>(d_z2, d_a2, hidden2_size);
        
        // Output layer
        // TODO: Final matrix multiplication
        
        // Softmax
        int blocks3 = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        softmax_kernel<<<blocks3, BLOCK_SIZE>>>(d_z3, d_output, output_size);
    }
public: 
    NeuralNetworkCUDA() {
        allocate_gpu_memory();
        initialize_weights();
    }

    ~NeuralNetworkCUDA() {
        // Free GPU memory
        cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_W3);
        cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_b3);
        cudaFree(d_input); cudaFree(d_z1); cudaFree(d_a1);
        cudaFree(d_z2); cudaFree(d_a2); cudaFree(d_z3);
        cudaFree(d_output); cudaFree(d_target);
        cudaFree(d_dZ3); cudaFree(d_dZ2); cudaFree(d_dZ1);
    }

    void train(const vector<vector<float>>& training_data, 
               const vector<int>& labels, 
               int epochs = 5, 
               float learning_rate = 0.01) {
        //TODO: Train here
    
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

bool loadFashionMNIST(vector<vector<float>>& images,
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
    images.resize(num_images, vector<float>(num_rows * num_cols));
    labels.resize(num_images);
    
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel = 0;
            file_images.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            if (file_images.fail()) {
                cerr << "Error reading image data." << endl;
                return false;
            }
            images[i][j] = pixel / 255.0; // Normalize to [0, 1]
        }

        unsigned char label = 0;
        file_labels.read(reinterpret_cast<char*>(&label), sizeof(label));
        if (file_labels.fail()) {
            cerr << "Error reading label data." << endl;
            return false;
        }
        labels[i] = static_cast<int>(label);
    }
    
    return true;
}

void printTrainingData(const vector<vector<float>>& images,
                       const vector<int>& labels, int count) {
    for (int i = 0; i < count && i < images.size(); ++i) {
        cout.precision(2);
        cout << "Sample " << i + 1 << ":\n";
        cout << "Label: " << labels[i] << "\n";
        cout << "Image (Flattened):\n";
        for (int j = 0; j < images[i].size(); ++j) {
            cout << images[i][j] << " ";
            if ((j + 1) % 28 == 0) cout << "\n"; // Format into 28 pixels per row
        }
        cout << "\n" << string(40, '-') << "\n";
    }
}

int main() {
    vector<vector<float>> images;
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

    // Convert vector<vector<float>> to Eigen::MatrixXd
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
    NeuralNetworkCUDA nn;
    cout << "Training neural network...\n";
    nn.train(training_data, training_labels);
    cout << "Training complete!\n";
    
    return 0;
}