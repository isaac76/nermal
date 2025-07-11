#include "neuralnetwork.h"
#include <QtTest/qtest.h>
#include <QObject>
#include <vector>
#include <cmath>

class TestNeuralNetwork : public QObject
{
    Q_OBJECT

private slots:
    void testBasicConstruction();
    void testBasicQueries();
    void testTrainingBasics();
    void testLearningRates();
};

void TestNeuralNetwork::testBasicConstruction()
{
    NeuralNetwork nn(3, 4, 2, 0.1);
    
    QCOMPARE(nn.getInputNodes(), 3);
    QCOMPARE(nn.getHiddenNodes(), 4);
    QCOMPARE(nn.getOutputNodes(), 2);
    QVERIFY(std::abs(nn.getLearningRate() - 0.1) < 1e-6);
}

void TestNeuralNetwork::testBasicQueries()
{
    NeuralNetwork nn(2, 3, 1, 0.5);
    
    std::vector<double> inputs = {0.5, 0.3};
    auto outputs = nn.query(inputs);
    
    QCOMPARE(outputs.size(), size_t(1));
    QVERIFY(outputs[0] > 0.0 && outputs[0] < 1.0);
}

void TestNeuralNetwork::testTrainingBasics()
{
    NeuralNetwork nn(2, 3, 1, 0.5);
    
    std::vector<double> inputs = {0.5, 0.3};
    std::vector<double> targets = {0.8};
    
    // Get initial output
    auto initial_output = nn.query(inputs);
    
    // Train multiple times
    for (int i = 0; i < 100; i++) {
        nn.train(inputs, targets);
    }
    
    // Get final output
    auto final_output = nn.query(inputs);
    
    // The output should have changed (learning occurred)
    QVERIFY(std::abs(final_output[0] - initial_output[0]) > 1e-6);
}

void TestNeuralNetwork::testLearningRates()
{
    NeuralNetwork nn(2, 4, 1, 0.5);
    
    // XOR training data
    std::vector<std::vector<double>> inputs = {
        {0.01, 0.01}, {0.01, 0.99}, {0.99, 0.01}, {0.99, 0.99}
    };
    std::vector<std::vector<double>> targets = {
        {0.01}, {0.99}, {0.99}, {0.01}
    };
    
    // Train for several epochs
    for (int epoch = 0; epoch < 1000; epoch++) {
        for (size_t i = 0; i < inputs.size(); i++) {
            nn.train(inputs[i], targets[i]);
        }
    }
    
    // Test the learned function
    auto result1 = nn.query({0.01, 0.01});  // Should be ~0.01
    auto result2 = nn.query({0.01, 0.99});  // Should be ~0.99
    auto result3 = nn.query({0.99, 0.01});  // Should be ~0.99
    auto result4 = nn.query({0.99, 0.99});  // Should be ~0.01
    
    // Check if learning occurred (outputs should be closer to targets)
    QVERIFY(result1[0] < 0.5);
    QVERIFY(result2[0] > 0.5);
    QVERIFY(result3[0] > 0.5);
    QVERIFY(result4[0] < 0.5);
}

QTEST_MAIN(TestNeuralNetwork)
#include "test_neuralnetwork.moc"
