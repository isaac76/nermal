#include "neuralnetwork.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>

/**
 * @brief Constructs a new Neural Network object
 * @param inputNodes Number of input nodes
 * @param hiddenNodes Number of hidden layer nodes
 * @param outputNodes Number of output nodes
 * @param learningRate Learning rate for training
 */
NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
    : inputNodes(inputNodes), hiddenNodes(hiddenNodes), outputNodes(outputNodes), learningRate(learningRate)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double stdInputToHidden = 1.0 / std::sqrt(inputNodes);
    std::normal_distribution<double> distInputToHidden(0.0, stdInputToHidden);
    
    // Weight matrix: each row is a hidden node, each column is an input node
    // Element (i,j) is the weight from input j to hidden node i
    weightsInputToHidden = Eigen::MatrixXd(hiddenNodes, inputNodes);
    for (int i = 0; i < hiddenNodes; ++i) {
        for (int j = 0; j < inputNodes; ++j) {
            weightsInputToHidden(i, j) = distInputToHidden(gen);
        }
    }
    
    double stdHiddenToOutput = 1.0 / std::sqrt(hiddenNodes);
    std::normal_distribution<double> distHiddenToOutput(0.0, stdHiddenToOutput);
    
    // Weight matrix: each row is an output node, each column is a hidden node
    // Element (i,j) is the weight from hidden node j to output node i
    weightsHiddenToOutput = Eigen::MatrixXd(outputNodes, hiddenNodes);
    for (int i = 0; i < outputNodes; ++i) {
        for (int j = 0; j < hiddenNodes; ++j) {
            weightsHiddenToOutput(i, j) = distHiddenToOutput(gen);
        }
    }
}

/**
 * @brief Sigmoid activation function: σ(x) = 1/(1 + e^(-x))
 * Maps any real number to (0,1) range. Has useful derivative: σ'(x) = σ(x)(1-σ(x))
 * which simplifies backpropagation calculations.
 */
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Matrix version of sigmoid function - applies sigmoid element-wise to entire matrix
 * Returns Eigen::MatrixXd because it processes multiple values at once (vectorized operation)
 * More efficient than calling scalar sigmoid in loops for neural network computations
 */
Eigen::MatrixXd NeuralNetwork::sigmoid(const Eigen::MatrixXd& matrix) {
    return matrix.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

/**
 * @brief Trains the neural network using backpropagation
 * @param inputsList Input data vector
 * @param targetsList Target output vector for supervised learning
 */
void NeuralNetwork::train(const std::vector<double>& inputsList, const std::vector<double>& targetsList) {
    // Convert input data to column matrix for matrix operations
    Eigen::MatrixXd inputs(inputNodes, 1);
    for (int i = 0; i < inputNodes; ++i) {
        inputs(i, 0) = inputsList[i];
    }
    
    // Convert target data to column matrix
    Eigen::MatrixXd targets(outputNodes, 1);
    for (int i = 0; i < outputNodes; ++i) {
        targets(i, 0) = targetsList[i];
    }
    
    // FORWARD PASS: Input layer → Hidden layer
    // Each hidden node receives weighted sum of ALL input nodes
    Eigen::MatrixXd hiddenInputs = weightsInputToHidden * inputs;
    Eigen::MatrixXd hiddenOutputs = sigmoid(hiddenInputs);  // Apply activation function
    
    // FORWARD PASS: Hidden layer → Output layer  
    // Each output node receives weighted sum of ALL hidden nodes
    Eigen::MatrixXd finalInputs = weightsHiddenToOutput * hiddenOutputs;
    Eigen::MatrixXd finalOutputs = sigmoid(finalInputs);   // Final predictions
    
    // BACKPROPAGATION: Calculate errors working backwards
    // Output error: how far off are our predictions?
    Eigen::MatrixXd outputErrors = targets - finalOutputs;
    
    // Hidden error: distribute output errors back to hidden nodes
    // Each hidden node's error depends on how much it contributed to output errors
    Eigen::MatrixXd hiddenErrors = weightsHiddenToOutput.transpose() * outputErrors;
    
    // UPDATE WEIGHTS: Hidden → Output layer
    // Gradient = error × sigmoid derivative × hidden node activation
    Eigen::MatrixXd outputGradients = outputErrors.cwiseProduct(finalOutputs).cwiseProduct(
        finalOutputs.unaryExpr([](double x) { return 1.0 - x; })  // Sigmoid derivative: σ(x)(1-σ(x))
    );
    // Adjust weights based on how much each hidden node contributed
    weightsHiddenToOutput += learningRate * outputGradients * hiddenOutputs.transpose();
    
    // UPDATE WEIGHTS: Input → Hidden layer
    // Gradient = error × sigmoid derivative × input node activation
    Eigen::MatrixXd hiddenGradients = hiddenErrors.cwiseProduct(hiddenOutputs).cwiseProduct(
        hiddenOutputs.unaryExpr([](double x) { return 1.0 - x; })  // Sigmoid derivative
    );
    // Weight update logic: "increase connection" means make weight more positive/less negative
    // If hiddenError > 0 (hidden node should have been MORE active): strengthen positive inputs
    // If hiddenError < 0 (hidden node should have been LESS active): weaken positive inputs
    // The direction depends on BOTH the error sign AND input value sign
    // inputs.transpose() creates matrix where each column represents input activation levels
    weightsInputToHidden += learningRate * hiddenGradients * inputs.transpose();
}

/**
 * @brief Performs a forward pass through the network to get predictions
 * @param inputsList Input data vector
 * @return std::vector<double> Network output predictions
 */
std::vector<double> NeuralNetwork::query(const std::vector<double>& inputsList) {
    Eigen::MatrixXd inputs(inputNodes, 1);
    for (int i = 0; i < inputNodes; ++i) {
        inputs(i, 0) = inputsList[i];
    }
    
    Eigen::MatrixXd hiddenInputs = weightsInputToHidden * inputs;
    Eigen::MatrixXd hiddenOutputs = sigmoid(hiddenInputs);
    
    Eigen::MatrixXd finalInputs = weightsHiddenToOutput * hiddenOutputs;
    Eigen::MatrixXd finalOutputs = sigmoid(finalInputs);
    
    std::vector<double> result(outputNodes);
    for (int i = 0; i < outputNodes; ++i) {
        result[i] = finalOutputs(i, 0);
    }
    
    return result;
}

/**
 * @brief Prints detailed information about the neural network structure
 */
void NeuralNetwork::printNetworkInfo() const {
    std::cout << "Neural Network Information:" << std::endl;
    std::cout << "  Input Nodes: " << inputNodes << std::endl;
    std::cout << "  Hidden Nodes: " << hiddenNodes << std::endl;
    std::cout << "  Output Nodes: " << outputNodes << std::endl;
    std::cout << "  Learning Rate: " << learningRate << std::endl;
    std::cout << "  Input-to-Hidden Weights Shape: " << weightsInputToHidden.rows() 
              << " x " << weightsInputToHidden.cols() << std::endl;
    std::cout << "  Hidden-to-Output Weights Shape: " << weightsHiddenToOutput.rows() 
              << " x " << weightsHiddenToOutput.cols() << std::endl;
}

std::vector<uint8_t> NeuralNetwork::serializeToBytes() const {
    std::vector<uint8_t> data;
    
    // Helper lambda to write data to byte vector
    auto writeBytes = [&data](const void* source, size_t size) {
        const uint8_t* bytes = static_cast<const uint8_t*>(source);
        data.insert(data.end(), bytes, bytes + size);
    };
    
    // Write magic number for format validation
    const uint32_t magic = 0x4E4E4448; // "NNDH" - Neural Network Data Header
    writeBytes(&magic, sizeof(magic));
    
    // Write version
    const uint32_t version = 1;
    writeBytes(&version, sizeof(version));
    
    // Write network configuration
    writeBytes(&inputNodes, sizeof(inputNodes));
    writeBytes(&hiddenNodes, sizeof(hiddenNodes));
    writeBytes(&outputNodes, sizeof(outputNodes));
    writeBytes(&learningRate, sizeof(learningRate));
    
    // Write input-to-hidden weights matrix dimensions
    int rows = weightsInputToHidden.rows();
    int cols = weightsInputToHidden.cols();
    writeBytes(&rows, sizeof(rows));
    writeBytes(&cols, sizeof(cols));
    
    // Write input-to-hidden weights data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double weight = weightsInputToHidden(i, j);
            writeBytes(&weight, sizeof(weight));
        }
    }
    
    // Write hidden-to-output weights matrix dimensions
    rows = weightsHiddenToOutput.rows();
    cols = weightsHiddenToOutput.cols();
    writeBytes(&rows, sizeof(rows));
    writeBytes(&cols, sizeof(cols));
    
    // Write hidden-to-output weights data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double weight = weightsHiddenToOutput(i, j);
            writeBytes(&weight, sizeof(weight));
        }
    }
    
    std::cout << "Neural network serialized to " << data.size() << " bytes" << std::endl;
    return data;
}

bool NeuralNetwork::deserializeFromBytes(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        std::cerr << "Error: Empty data provided for deserialization" << std::endl;
        return false;
    }
    
    size_t offset = 0;
    
    // Helper lambda to read data from byte vector
    auto readBytes = [&data, &offset](void* dest, size_t size) -> bool {
        if (offset + size > data.size()) {
            return false;
        }
        std::memcpy(dest, &data[offset], size);
        offset += size;
        return true;
    };
    
    try {
        // Read and validate magic number
        uint32_t magic;
        if (!readBytes(&magic, sizeof(magic)) || magic != 0x4E4E4448) {
            std::cerr << "Error: Invalid file format or corrupted data" << std::endl;
            return false;
        }
        
        // Read version
        uint32_t version;
        if (!readBytes(&version, sizeof(version)) || version != 1) {
            std::cerr << "Error: Unsupported file version: " << version << std::endl;
            return false;
        }
        
        // Read network configuration
        int newInputNodes, newHiddenNodes, newOutputNodes;
        double newLearningRate;
        
        if (!readBytes(&newInputNodes, sizeof(newInputNodes)) ||
            !readBytes(&newHiddenNodes, sizeof(newHiddenNodes)) ||
            !readBytes(&newOutputNodes, sizeof(newOutputNodes)) ||
            !readBytes(&newLearningRate, sizeof(newLearningRate))) {
            std::cerr << "Error: Failed to read network configuration" << std::endl;
            return false;
        }
        
        // Validate configuration matches current network
        if (newInputNodes != inputNodes || newHiddenNodes != hiddenNodes || 
            newOutputNodes != outputNodes) {
            std::cerr << "Error: Network configuration mismatch!" << std::endl;
            std::cerr << "File: " << newInputNodes << "x" << newHiddenNodes << "x" << newOutputNodes << std::endl;
            std::cerr << "Current: " << inputNodes << "x" << hiddenNodes << "x" << outputNodes << std::endl;
            return false;
        }
        
        // Update learning rate
        learningRate = newLearningRate;
        
        // Read input-to-hidden weights
        int rows, cols;
        if (!readBytes(&rows, sizeof(rows)) || !readBytes(&cols, sizeof(cols))) {
            std::cerr << "Error: Failed to read input-to-hidden matrix dimensions" << std::endl;
            return false;
        }
        
        if (rows != hiddenNodes || cols != inputNodes) {
            std::cerr << "Error: Input-to-hidden matrix dimension mismatch" << std::endl;
            return false;
        }
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double weight;
                if (!readBytes(&weight, sizeof(weight))) {
                    std::cerr << "Error: Failed to read input-to-hidden weights" << std::endl;
                    return false;
                }
                weightsInputToHidden(i, j) = weight;
            }
        }
        
        // Read hidden-to-output weights
        if (!readBytes(&rows, sizeof(rows)) || !readBytes(&cols, sizeof(cols))) {
            std::cerr << "Error: Failed to read hidden-to-output matrix dimensions" << std::endl;
            return false;
        }
        
        if (rows != outputNodes || cols != hiddenNodes) {
            std::cerr << "Error: Hidden-to-output matrix dimension mismatch" << std::endl;
            return false;
        }
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double weight;
                if (!readBytes(&weight, sizeof(weight))) {
                    std::cerr << "Error: Failed to read hidden-to-output weights" << std::endl;
                    return false;
                }
                weightsHiddenToOutput(i, j) = weight;
            }
        }
        
        std::cout << "Neural network deserialized successfully from " << data.size() << " bytes" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error deserializing neural network: " << e.what() << std::endl;
        return false;
    }
}