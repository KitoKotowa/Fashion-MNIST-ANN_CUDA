#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>

using namespace std;

// Data loading function (use your existing implementation)
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
    
    for(int i = 0; i < num_images; i++) {
        for(int j = 0; j < num_rows * num_cols; j++) {
            unsigned char pixel = 0;
            file_images.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = pixel / 255.0;  // Normalize to [0,1]
        }
        
        unsigned char label = 0;
        file_labels.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    
    return true;
}

int main() {
    vector<vector<double>> training_data;
    vector<int> training_labels;
    
    string image_file = "train-images-idx3-ubyte";
    string label_file = "train-labels-idx1-ubyte";
    
    if (!loadFashionMNIST(training_data, training_labels, image_file, label_file)) {
        cerr << "Failed to load Fashion MNIST dataset" << endl;
        return 1;
    }
    
    // NeuralNetwork nn;
    // nn.train(training_data, training_labels);
    
    return 0;
}