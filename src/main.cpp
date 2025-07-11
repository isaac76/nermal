#include "mainwindow.h"
#include "neuralnetwork.h"

#include <QApplication>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    // Test the neural network
    std::cout << "Testing Neural Network..." << std::endl;
    
    // Create a simple network (3 input, 3 hidden, 1 output)
    NeuralNetwork nn(3, 3, 1, 0.3);
    nn.printNetworkInfo();
    
    // Test with some sample data
    std::vector<double> inputs = {0.5, 0.8, 0.2};
    std::vector<double> targets = {0.9};
    
    std::cout << "\nBefore training:" << std::endl;
    std::vector<double> result = nn.query(inputs);
    std::cout << "Input: [0.5, 0.8, 0.2], Output: " << result[0] << std::endl;
    
    // Train the network
    for (int i = 0; i < 1000; ++i) {
        nn.train(inputs, targets);
    }
    
    std::cout << "\nAfter 1000 training iterations:" << std::endl;
    result = nn.query(inputs);
    std::cout << "Input: [0.5, 0.8, 0.2], Output: " << result[0] 
              << " (Target: " << targets[0] << ")" << std::endl;
    
    MainWindow w;
    w.show();
    return a.exec();
}
