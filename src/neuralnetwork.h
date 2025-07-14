#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <memory>
#include <cstdint>

class NeuralNetwork
{
private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;

    // Weight matrices using Eigen
    Eigen::MatrixXd weightsInputToHidden;
    Eigen::MatrixXd weightsHiddenToOutput;

    // Sigmoid activation function
    static double sigmoid(double x);
    static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& matrix);

public:
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate);
    
    // Train the network with input and target data
    void train(const std::vector<double>& inputsList, const std::vector<double>& targetsList);
    
    // Query the network (forward pass)
    std::vector<double> query(const std::vector<double>& inputsList);

    // Serialize network data to binary format
    std::vector<uint8_t> serializeToBytes() const;

    // Deserialize network data from binary format
    bool deserializeFromBytes(const std::vector<uint8_t>& data);
    
    // Print network information
    void printNetworkInfo() const;
    
    // Getters
    int getInputNodes() const { return inputNodes; }
    int getHiddenNodes() const { return hiddenNodes; }
    int getOutputNodes() const { return outputNodes; }
    double getLearningRate() const { return learningRate; }
};

#endif // NEURALNETWORK_H
