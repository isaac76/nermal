#include "neuralnetwork.h"
#include <random>
#include <cmath>
#include <algorithm>

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
    : inputNodes(inputNodes), hiddenNodes(hiddenNodes), outputNodes(outputNodes), learningRate(learningRate)
{
    // Initialize weight matrices with random values
    // Using normal distribution with mean=0.0 and std=1/sqrt(number_of_input_nodes)
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Initialize weights from input to hidden layer
    double stdInputToHidden = 1.0 / std::sqrt(inputNodes);
    std::normal_distribution<double> distInputToHidden(0.0, stdInputToHidden);
    
    weightsInputToHidden = Eigen::MatrixXd(hiddenNodes, inputNodes);
    for (int i = 0; i < hiddenNodes; ++i) {
        for (int j = 0; j < inputNodes; ++j) {
            weightsInputToHidden(i, j) = distInputToHidden(gen);
        }
    }
    
    // Initialize weights from hidden to output layer
    double stdHiddenToOutput = 1.0 / std::sqrt(hiddenNodes);
    std::normal_distribution<double> distHiddenToOutput(0.0, stdHiddenToOutput);
    
    weightsHiddenToOutput = Eigen::MatrixXd(outputNodes, hiddenNodes);
    for (int i = 0; i < outputNodes; ++i) {
        for (int j = 0; j < hiddenNodes; ++j) {
            weightsHiddenToOutput(i, j) = distHiddenToOutput(gen);
        }
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

Eigen::MatrixXd NeuralNetwork::sigmoid(const Eigen::MatrixXd& matrix) {
    return matrix.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

void NeuralNetwork::train(const std::vector<double>& inputsList, const std::vector<double>& targetsList) {
    // Convert input vectors to Eigen matrices
    Eigen::MatrixXd inputs(inputNodes, 1);
    for (int i = 0; i < inputNodes; ++i) {
        inputs(i, 0) = inputsList[i];
    }
    
    Eigen::MatrixXd targets(outputNodes, 1);
    for (int i = 0; i < outputNodes; ++i) {
        targets(i, 0) = targetsList[i];
    }
    
    // Forward pass
    Eigen::MatrixXd hiddenInputs = weightsInputToHidden * inputs;
    Eigen::MatrixXd hiddenOutputs = sigmoid(hiddenInputs);
    
    Eigen::MatrixXd finalInputs = weightsHiddenToOutput * hiddenOutputs;
    Eigen::MatrixXd finalOutputs = sigmoid(finalInputs);
    
    // Calculate errors
    Eigen::MatrixXd outputErrors = targets - finalOutputs;
    Eigen::MatrixXd hiddenErrors = weightsHiddenToOutput.transpose() * outputErrors;
    
    // Update weights using backpropagation
    // Update weights from hidden to output
    Eigen::MatrixXd outputGradients = outputErrors.cwiseProduct(finalOutputs).cwiseProduct(
        finalOutputs.unaryExpr([](double x) { return 1.0 - x; })
    );
    weightsHiddenToOutput += learningRate * outputGradients * hiddenOutputs.transpose();
    
    // Update weights from input to hidden
    Eigen::MatrixXd hiddenGradients = hiddenErrors.cwiseProduct(hiddenOutputs).cwiseProduct(
        hiddenOutputs.unaryExpr([](double x) { return 1.0 - x; })
    );
    weightsInputToHidden += learningRate * hiddenGradients * inputs.transpose();
}

std::vector<double> NeuralNetwork::query(const std::vector<double>& inputsList) {
    // Convert input vector to Eigen matrix
    Eigen::MatrixXd inputs(inputNodes, 1);
    for (int i = 0; i < inputNodes; ++i) {
        inputs(i, 0) = inputsList[i];
    }
    
    // Forward pass
    Eigen::MatrixXd hiddenInputs = weightsInputToHidden * inputs;
    Eigen::MatrixXd hiddenOutputs = sigmoid(hiddenInputs);
    
    Eigen::MatrixXd finalInputs = weightsHiddenToOutput * hiddenOutputs;
    Eigen::MatrixXd finalOutputs = sigmoid(finalInputs);
    
    // Convert result back to std::vector
    std::vector<double> result(outputNodes);
    for (int i = 0; i < outputNodes; ++i) {
        result[i] = finalOutputs(i, 0);
    }
    
    return result;
}

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
