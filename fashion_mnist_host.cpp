#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace std;

class NeuralNetwork {
private:
    static const int input_size = 784;
    static const int hidden1_size = 128;
    static const int hidden2_size = 128;
    static const int output_size = 10;

    Eigen::MatrixXd W1; // Input to first hidden layer
    Eigen::MatrixXd W2; // First to second hidden layer
    Eigen::MatrixXd W3; // Second hidden layer to output

    // Bias vectors
    Eigen::VectorXd b1; // Bias for first hidden layer
    Eigen::VectorXd b2; // Bias for second hidden layer
    Eigen::VectorXd b3; // Bias for output layer

    // Activation functions
    Eigen::VectorXd ReLU(const Eigen::VectorXd& x) {
        return x.cwiseMax(0.0);
    }

    Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
        Eigen::VectorXd exp_x = x.array().exp();
        return exp_x / exp_x.sum();
    }

    Eigen::VectorXd ReLU_derivative(const Eigen::VectorXd& x) {
        return (x.array() > 0).cast<double>();
    }

    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0, 0.01);

        // Xavier/Glorot initialization
        W1 = Eigen::MatrixXd::NullaryExpr(hidden1_size, input_size, 
            [&]() { return d(gen) * sqrt(2.0 / (input_size + hidden1_size)); });
        W2 = Eigen::MatrixXd::NullaryExpr(hidden2_size, hidden1_size, 
            [&]() { return d(gen) * sqrt(2.0 / (hidden1_size + hidden2_size)); });
        W3 = Eigen::MatrixXd::NullaryExpr(output_size, hidden2_size, 
            [&]() { return d(gen) * sqrt(2.0 / (hidden2_size + output_size)); });

        b1 = Eigen::VectorXd::Zero(hidden1_size);
        b2 = Eigen::VectorXd::Zero(hidden2_size);
        b3 = Eigen::VectorXd::Zero(output_size);
    }

public:
    NeuralNetwork() {
        initialize_weights();
    }

    Eigen::VectorXd forward_propagation(const Eigen::VectorXd& input) {
        // First hidden layer
        Eigen::VectorXd z1 = W1 * input + b1;
        Eigen::VectorXd a1 = ReLU(z1);

        // Second hidden layer
        Eigen::VectorXd z2 = W2 * a1 + b2;
        Eigen::VectorXd a2 = ReLU(z2);

        // Output layer
        Eigen::VectorXd z3 = W3 * a2 + b3;
        Eigen::VectorXd output = softmax(z3);

        return output;
    }

    void train(const vector<vector<double>>& training_data, 
               const vector<int>& labels, 
               int epochs = 10, 
               double learning_rate = 0.01) {

        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();

            double total_loss = 0.0;
            int correct_pred = 0;

            for (size_t i = 0; i < training_data.size(); ++i) {
                Eigen::VectorXd input = Eigen::Map<const Eigen::VectorXd>(training_data[i].data(), input_size);

                // One-hot encoded label
                Eigen::VectorXd target = Eigen::VectorXd::Zero(output_size);
                target(labels[i]) = 1.0;

                // Forward Propagation
                Eigen::VectorXd z1 = W1 * input + b1;
                Eigen::VectorXd a1 = ReLU(z1);

                Eigen::VectorXd z2 = W2 * a1 + b2;
                Eigen::VectorXd a2 = ReLU(z2);

                Eigen::VectorXd z3 = W3 * a2 + b3;
                Eigen::VectorXd output = softmax(z3);

                // Compute loss (Cross-Entropy)
                double loss = -target.dot(output.array().log().matrix());
                total_loss += loss;

                // Backward Propagation
                // Output layer gradient
                Eigen::VectorXd dZ3 = output - target;

                // Second hidden layer gradients
                Eigen::MatrixXd dW3 = dZ3 * a2.transpose();
                Eigen::VectorXd db3 = dZ3;

                Eigen::VectorXd dZ2 = (W3.transpose() * dZ3).array() * ReLU_derivative(z2).array();
                Eigen::MatrixXd dW2 = dZ2 * a1.transpose();
                Eigen::VectorXd db2 = dZ2;

                // First hidden layer gradients
                Eigen::VectorXd dZ1 = (W2.transpose() * dZ2).array() * ReLU_derivative(z1).array();
                Eigen::MatrixXd dW1 = dZ1 * input.transpose();
                Eigen::VectorXd db1 = dZ1;

                // Update weights and biases
                W3 -= learning_rate * dW3;
                b3 -= learning_rate * db3;
                W2 -= learning_rate * dW2;
                b2 -= learning_rate * db2;
                W1 -= learning_rate * dW1;
                b1 -= learning_rate * db1;

                int predicted_class;
                output.maxCoeff(&predicted_class);
                if (predicted_class == labels[i]) {
                    correct_pred++;
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);

            cout << "Epoch " << epoch + 1 
                    << ", Average Loss: " << total_loss / training_data.size() 
                    << ", Accuracy: " << static_cast<double>(correct_pred) / training_data.size() 
                    << ", Epoch Time: " << epoch_duration.count() << " ms"
                    << endl;
        }
    }


    int predict(const vector<double>& input) {
        Eigen::VectorXd eigen_input = Eigen::Map<const Eigen::VectorXd>(input.data(), input_size);
        Eigen::VectorXd output = forward_propagation(eigen_input);
        
        // Find the index of the maximum probability
        int predicted_class;
        output.maxCoeff(&predicted_class);
        return predicted_class;
    }
};

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
    
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file_images.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file_images.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file_images.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file_images.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    
    file_labels.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file_labels.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    
    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
    
    images.resize(num_images, vector<double>(num_rows * num_cols));
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
    vector<vector<double>> images;
    vector<int> labels;
    
    const string image_file = "train-images-idx3-ubyte";
    const string label_file = "train-labels-idx1-ubyte";
    
    if (loadFashionMNIST(images, labels, image_file, label_file)) {
        cout << "Loaded " << images.size() << " training samples.\n";
        // cout << "Printing the first 20 samples:\n";
        // printTrainingData(images, labels, 20);
        NeuralNetwork nn;
        nn.batch_train(images, labels);
        nn.train(images, labels);

    } else {
        cout << "Failed to load Fashion-MNIST dataset." << endl;
    }

    return 0;
}
