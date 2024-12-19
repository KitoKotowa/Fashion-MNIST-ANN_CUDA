#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

using namespace std;

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << endl; \
        exit(1); \
    } \
}

#define BLOCK_SIZE 256

int find_max_element_index(const std::vector<float>& h_output) {
    float max_val = h_output[0];
    int max_idx = 0;
    for (int i = 1; i < h_output.size(); ++i) {
        if (h_output[i] > max_val) {
            max_val = h_output[i];
            max_idx = i;
        }
    }
    return max_idx;
}

__global__ void softmax_kernel(float* input, float* output, int size) {
    __shared__ float max_val;
    __shared__ float sum;

    if (threadIdx.x == 0) {
        max_val = input[0];
        for  (int i = 1; i < size; ++i) {
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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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

__global__ void matrix_multiply_kernel(float* A, float* B, float* C, float* bias, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float value = 0.0f;
        for (int e = 0; e < N; ++e) {
            value += A[row * N + e] * B[e * K + col];
        }
        C[row * K + col] = value + bias[row];
    }
}

__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

void forward_propagation_kernel(float* input, float* W1, float* b1, float* W2, float* b2, float* W3, float* b3, float* z1, float* a1, float* z2, float* a2, float* z3, float* output, int input_size, int hidden1_size, int hidden2_size, int output_size) {
        // First hidden layer
        dim3 blockDim(16, 16);
        dim3 gridDim1((input_size + blockDim.x - 1) / blockDim.x, 
                    (hidden1_size + blockDim.y - 1) / blockDim.y);
        
        matrix_multiply_kernel<<<gridDim1, blockDim>>>(W1, input, z1, b1, hidden1_size, input_size, 1);
        
        // ReLU activation
        int blocks1 = (hidden1_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relu_kernel<<<blocks1, BLOCK_SIZE>>>(z1, a1, hidden1_size);
        
        // Second hidden layer
        dim3 gridDim2((hidden1_size + blockDim.x - 1) / blockDim.x,
                    (hidden2_size + blockDim.y - 1) / blockDim.y);
        
        matrix_multiply_kernel<<<gridDim2, blockDim>>>(W2, a1, z2, b2, hidden2_size, hidden1_size, 1);
        
        // ReLU activation
        int blocks2 = (hidden2_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relu_kernel<<<blocks2, BLOCK_SIZE>>>(z2, a2, hidden2_size);
        
        // Output layer
        dim3 gridDim3((hidden2_size + blockDim.x - 1) / blockDim.x,
                    (output_size + blockDim.y - 1) / blockDim.y);
        
        matrix_multiply_kernel<<<gridDim3, blockDim>>>(W3, a2, z3, b3, output_size, hidden2_size, 1);
        
        // Softmax
        int blocks3 = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        softmax_kernel<<<blocks3, BLOCK_SIZE>>>(z3, output, output_size);
    }

__global__ void backward_propagation_kernel(
    float *input, float *target, float *W1, float *b1, float *W2, float *b2, float *W3, float *b3,
    float *output, float *hidden1, float *hidden2, float learning_rate, 
    int input_size, int hidden_size1, int hidden_size2, int output_size) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float shared_mem[];
    float *delta_output = &shared_mem[0];       
    float *delta_hidden2 = &shared_mem[output_size];
    float *delta_hidden1 = &shared_mem[output_size + hidden_size2];

    // Compute output layer error
    if (idx < output_size) {
        delta_output[idx] = output[idx] - target[idx];
    }
    __syncthreads();

    // Backpropagate error to second hidden layer
    if (idx < hidden_size2) {
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            sum += W3[i * hidden_size2 + idx] * delta_output[i];
        }
        delta_hidden2[idx] = (hidden2[idx] > 0.0f) ? sum : 0.0f; // ReLU derivative
    }
    __syncthreads();

    // Backpropagate error to first hidden layer
    if (idx < hidden_size1) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size2; i++) {
            sum += W2[i * hidden_size1 + idx] * delta_hidden2[i];
        }
        delta_hidden1[idx] = (hidden1[idx] > 0.0f) ? sum : 0.0f; // ReLU derivative
    }
    __syncthreads();

    // Update weights and biases for W3 and b3
    if (idx < output_size) {
        for (int i = 0; i < hidden_size2; i++) {
            W3[idx * hidden_size2 + i] -= learning_rate * delta_output[idx] * hidden2[i];
        }
        b3[idx] -= learning_rate * delta_output[idx];
    }
    __syncthreads();

    // Update weights and biases for W2 and b2
    if (idx < hidden_size2) {
        for (int i = 0; i < hidden_size1; i++) {
            W2[idx * hidden_size1 + i] -= learning_rate * delta_hidden2[idx] * hidden1[i];
        }
        b2[idx] -= learning_rate * delta_hidden2[idx];
    }
    __syncthreads();

    // Update weights and biases for W1 and b1
    if (idx < hidden_size1) {
        for (int i = 0; i < input_size; i++) {
            W1[idx * input_size + i] -= learning_rate * delta_hidden1[idx] * input[i];
        }
        b1[idx] -= learning_rate * delta_hidden1[idx];
    }
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
        CHECK(cudaMalloc(&d_W1, hidden1_size * input_size * sizeof(float)));
        CHECK(cudaMalloc(&d_W2, hidden2_size * hidden1_size * sizeof(float)));
        CHECK(cudaMalloc(&d_W3, output_size * hidden2_size * sizeof(float)));
        
        CHECK(cudaMalloc(&d_b1, hidden1_size * sizeof(float)));   
        CHECK(cudaMalloc(&d_b2, hidden2_size * sizeof(float)));
        CHECK(cudaMalloc(&d_b3, output_size * sizeof(float)));

        // Intermediate layers
        CHECK(cudaMalloc(&d_z1, hidden1_size * sizeof(float)));
        CHECK(cudaMalloc(&d_a1, hidden1_size * sizeof(float)));
        CHECK(cudaMalloc(&d_z2, hidden2_size * sizeof(float)));
        CHECK(cudaMalloc(&d_a2, hidden2_size * sizeof(float)));
        CHECK(cudaMalloc(&d_z3, output_size * sizeof(float)));
        CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
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

        CHECK(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_W3, h_W3.data(), h_W3.size() * sizeof(float), cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

public:
    NeuralNetworkCUDA() {
        allocate_memory();
        initialize_weights();
    }

    ~NeuralNetworkCUDA() {
        cudaFree(d_W1);
        cudaFree(d_W2);
        cudaFree(d_W3);
        cudaFree(d_b1);
        cudaFree(d_b2);
        cudaFree(d_b3);
        cudaFree(d_z1);
        cudaFree(d_a1);
        cudaFree(d_z2);
        cudaFree(d_a2);
        cudaFree(d_z3);
        cudaFree(d_output);
    }

    void train(const vector<vector<float>>& training_data, const vector<int>& labels, int epochs = 10) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = chrono::high_resolution_clock::now();

            double total_loss = 0.0;
            int correct_pred = 0;

            for (size_t i = 0; i < training_data.size(); ++i) {
                float *d_input, *d_target;
                CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
                CHECK(cudaMalloc(&d_target, output_size * sizeof(float)));

                CHECK(cudaMemcpy(d_input, training_data[i].data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

                vector<float> h_target(output_size, 0.0f);
                h_target[labels[i]] = 1.0f;
                CHECK(cudaMemcpy(d_target, h_target.data(), output_size * sizeof(float), cudaMemcpyHostToDevice));

                dim3 blockDim(BLOCK_SIZE);
                dim3 gridDim1((hidden1_size + blockDim.x - 1) / blockDim.x);
                dim3 gridDim2((hidden2_size + blockDim.x - 1) / blockDim.x);
                dim3 gridDim3((output_size + blockDim.x - 1) / blockDim.x);
                
                forward_propagation_kernel(d_input, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_z1, d_a1, d_z2, d_a2, d_z3, d_output, input_size, hidden1_size, hidden2_size, output_size);
                backward_propagation_kernel<<<gridDim3, blockDim>>>(d_input, d_target, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_output, d_a1, d_a2, learning_rate, input_size, hidden1_size, hidden2_size, output_size);

                cudaFree(d_input);
                cudaFree(d_target);
            }

            auto epoch_end = chrono::high_resolution_clock::now();
            auto epoch_duration = chrono::duration_cast<chrono::milliseconds>(epoch_end - epoch_start);

            cout << "Epoch " << epoch + 1 
                      << ", Average Loss: " << total_loss / training_data.size() 
                      << ", Accuracy: " << static_cast<double>(correct_pred) / training_data.size() 
                      << ", Epoch Time: " << epoch_duration.count() << " ms"
                      << endl;
        }
    }

    // int predict(const vector<float>& input) {
    //     float *d_input, *d_output;
    //     CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    //     CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    //     CHECK(cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

    //     forward_propagation_kernel<<<1, 1>>>(d_input, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_z1, d_a1, d_z2, d_a2, d_z3, d_output, input_size, hidden1_size, hidden2_size, output_size);

    //     vector<float> h_output(output_size);
    //     CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    //     int predicted_class = find_max_element_index(h_output.begin(), h_output.end()) - h_output.begin();

    //     cudaFree(d_input);
    //     cudaFree(d_output);

    //     return predicted_class;
    // }

    int predict(const std::vector<float>& input) {
        float *d_input, *d_output;
        CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
        CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

        CHECK(cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

        // // Configure grid and block dimensions
        // dim3 blockDim(BLOCK_SIZE);
        // dim3 gridDim1((hidden1_size + blockDim.x - 1) / blockDim.x);
        // dim3 gridDim2((hidden2_size + blockDim.x - 1) / blockDim.x);
        // dim3 gridDim3((output_size + blockDim.x - 1) / blockDim.x);

        forward_propagation_kernel(d_input, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3, d_z1, d_a1, d_z2, d_a2, d_z3, d_output, input_size, hidden1_size, hidden2_size, output_size);

        std::vector<float> h_output(output_size);
        CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

        int predicted_class = find_max_element_index(h_output);

        cudaFree(d_input);
        cudaFree(d_output);

        return predicted_class;
    }
};

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

int main() {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    
    const std::string image_file = "train-images-idx3-ubyte";
    const std::string label_file = "train-labels-idx1-ubyte";
    
    if (loadFashionMNIST(images, labels, image_file, label_file)) {
        std::cout << "Loaded " << images.size() << " training samples.\n";
        NeuralNetworkCUDA nn;
        nn.train(images, labels);
    } else {
        std::cout << "Failed to load Fashion-MNIST dataset." << std::endl;
    }

    return 0;
}