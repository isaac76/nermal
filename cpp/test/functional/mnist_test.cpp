#include "neuralnetwork.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Helper function to split CSV line
std::vector<std::string> split(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Helper function to normalize pixel values (0-255 to 0.01-0.99)
std::vector<double> normalizePixels(const std::vector<std::string>& pixelStrings) {
    std::vector<double> normalized;
    normalized.reserve(pixelStrings.size());
    
    for (const auto& pixel : pixelStrings) {
        double value = std::stod(pixel);
        // Normalize from 0-255 to 0.01-0.99 (matching Python implementation)
        normalized.push_back((value / 255.0 * 0.99) + 0.01);
    }
    
    return normalized;
}

// Helper function to create target vector
std::vector<double> createTargets(int label, int outputNodes) {
    std::vector<double> targets(outputNodes, 0.01);
    targets[label] = 0.99;
    return targets;
}

// Function to load and process training data
std::vector<std::pair<std::vector<double>, std::vector<double>>> loadTrainingData(const std::string& filename, int maxSamples = -1) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return trainingData;
    }
    
    int count = 0;
    while (std::getline(file, line) && (maxSamples == -1 || count < maxSamples)) {
        auto tokens = split(line, ',');
        if (tokens.size() != 785) { // 1 label + 784 pixels
            std::cerr << "Invalid line format, expected 785 values, got " << tokens.size() << std::endl;
            continue;
        }
        
        int label = std::stoi(tokens[0]);
        std::vector<std::string> pixelStrings(tokens.begin() + 1, tokens.end());
        
        auto inputs = normalizePixels(pixelStrings);
        auto targets = createTargets(label, 10);
        
        trainingData.emplace_back(inputs, targets);
        count++;
    }
    
    file.close();
    std::cout << "Loaded " << trainingData.size() << " training samples" << std::endl;
    return trainingData;
}

// Function to load test data
std::vector<std::pair<std::vector<double>, int>> loadTestData(const std::string& filename, int maxSamples = -1) {
    std::vector<std::pair<std::vector<double>, int>> testData;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return testData;
    }
    
    int count = 0;
    while (std::getline(file, line) && (maxSamples == -1 || count < maxSamples)) {
        auto tokens = split(line, ',');
        if (tokens.size() != 785) {
            std::cerr << "Invalid line format, expected 785 values, got " << tokens.size() << std::endl;
            continue;
        }
        
        int label = std::stoi(tokens[0]);
        std::vector<std::string> pixelStrings(tokens.begin() + 1, tokens.end());
        
        auto inputs = normalizePixels(pixelStrings);
        
        testData.emplace_back(inputs, label);
        count++;
    }
    
    file.close();
    std::cout << "Loaded " << testData.size() << " test samples" << std::endl;
    return testData;
}

// Function to test the network accuracy
double testNetworkAccuracy(NeuralNetwork& network, const std::vector<std::pair<std::vector<double>, int>>& testData, int maxSamples = -1) {
    int correct = 0;
    int total = 0;
    
    int limit = (maxSamples == -1) ? testData.size() : std::min(maxSamples, (int)testData.size());
    
    for (int i = 0; i < limit; i++) {
        const auto& sample = testData[i];
        auto result = network.query(sample.first);
        
        // Find the index of the maximum output (predicted digit)
        int predicted = std::max_element(result.begin(), result.end()) - result.begin();
        
        if (predicted == sample.second) {
            correct++;
        }
        total++;
    }
    
    return (double)correct / total;
}

int main() {
    std::cout << "=== MNIST Neural Network Training and Testing ===" << std::endl;
    
    // Network parameters (matching Python implementation)
    int inputNodes = 784;   // 28x28 pixels
    int hiddenNodes = 100;  // Hidden layer size
    int outputNodes = 10;   // 10 digits (0-9)
    double learningRate = 0.3;
    
    // Determine sample sizes based on build configuration
#ifdef QUICK_TEST
    int trainingSamples = 100;
    int testSamples = 20;
    int epochs = 2;
    std::cout << "Running QUICK TEST mode" << std::endl;
#else
    int trainingSamples = 1000;
    int testSamples = 100;
    int epochs = 5;
    std::cout << "Running FULL TEST mode" << std::endl;
#endif
    
    // Create neural network
    NeuralNetwork nermal(inputNodes, hiddenNodes, outputNodes, learningRate);
    nermal.printNetworkInfo();
    
    // Load training data
    std::cout << "\n=== Loading Training Data ===" << std::endl;
    auto trainingData = loadTrainingData("csv/mnist_train.csv", trainingSamples);
    
    if (trainingData.empty()) {
        std::cerr << "No training data loaded. Exiting." << std::endl;
        return 1;
    }
    
    // Train the network
    std::cout << "\n=== Training Network ===" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << ": ";
        std::cout.flush();
        
        // Shuffle training data
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(trainingData.begin(), trainingData.end(), g);
        
        for (const auto& sample : trainingData) {
            nermal.train(sample.first, sample.second);
        }
        
        std::cout << "Complete" << std::endl;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Training completed in " << duration.count() << " ms" << std::endl;
    
    // Load test data
    std::cout << "\n=== Loading Test Data ===" << std::endl;
    auto testData = loadTestData("csv/mnist_test.csv", testSamples);
    
    if (testData.empty()) {
        std::cerr << "No test data loaded. Exiting." << std::endl;
        return 1;
    }
    
    // Test the network
    std::cout << "\n=== Testing Network ===" << std::endl;
    double accuracy = testNetworkAccuracy(nermal, testData);
    std::cout << "Accuracy: " << (accuracy * 100) << "%" << std::endl;
    
    // Test individual samples (like Python version)
    std::cout << "\n=== Individual Sample Testing ===" << std::endl;
    int sampleCount = std::min(5, (int)testData.size());
    for (int i = 0; i < sampleCount; i++) {
        const auto& sample = testData[i];
        auto result = nermal.query(sample.first);
        
        int predicted = std::max_element(result.begin(), result.end()) - result.begin();
        double confidence = *std::max_element(result.begin(), result.end());
        
        std::cout << "Sample " << i + 1 << ":" << std::endl;
        std::cout << "  Actual digit: " << sample.second << std::endl;
        std::cout << "  Predicted digit: " << predicted << std::endl;
        std::cout << "  Confidence: " << confidence << std::endl;
        std::cout << "  Result: " << (predicted == sample.second ? "CORRECT" : "INCORRECT") << std::endl;
        
        // Show confidence for each digit (only in full test mode)
#ifndef QUICK_TEST
        std::cout << "  Confidence for each digit:" << std::endl;
        for (int j = 0; j < result.size(); j++) {
            std::cout << "    Digit " << j << ": " << result[j] << std::endl;
        }
#endif
    }
    
    return 0;
}
