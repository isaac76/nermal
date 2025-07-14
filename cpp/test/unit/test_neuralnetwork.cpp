#include "neuralnetwork.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// Test fixture for NeuralNetwork tests
class NeuralNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code that runs before each test
    }

    void TearDown() override {
        // Cleanup code that runs after each test
    }
};

TEST_F(NeuralNetworkTest, BasicConstruction) {
    NeuralNetwork nn(3, 4, 2, 0.1);
    
    EXPECT_EQ(nn.getInputNodes(), 3);
    EXPECT_EQ(nn.getHiddenNodes(), 4);
    EXPECT_EQ(nn.getOutputNodes(), 2);
    EXPECT_NEAR(nn.getLearningRate(), 0.1, 1e-6);
}

TEST_F(NeuralNetworkTest, BasicQueries) {
    NeuralNetwork nn(2, 3, 1, 0.5);
    
    std::vector<double> inputs = {0.5, 0.3};
    auto outputs = nn.query(inputs);
    
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_GT(outputs[0], 0.0);
    EXPECT_LT(outputs[0], 1.0);
}

TEST_F(NeuralNetworkTest, TrainingBasics) {
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
    EXPECT_GT(std::abs(final_output[0] - initial_output[0]), 1e-6);
}

TEST_F(NeuralNetworkTest, LearningProgress) {
    NeuralNetwork nn(2, 4, 1, 0.3);
    
    // Simple training data - one input should map to high output
    std::vector<double> inputs = {0.9, 0.1};
    std::vector<double> targets = {0.9};
    
    // Get initial performance
    auto initial_output = nn.query(inputs);
    
    // Train for several iterations
    for (int i = 0; i < 500; i++) {
        nn.train(inputs, targets);
    }
    
    // Get final performance
    auto final_output = nn.query(inputs);
    
    // Test that learning occurred (output should be closer to target)
    double initial_error = std::abs(initial_output[0] - targets[0]);
    double final_error = std::abs(final_output[0] - targets[0]);
    
    // The final error should be less than the initial error (learning occurred)
    EXPECT_LT(final_error, initial_error);
    
    // The network should have learned something significant
    EXPECT_GT(initial_error - final_error, 0.01);
}

TEST_F(NeuralNetworkTest, Serialization) {
    // Create and train a network
    NeuralNetwork original(3, 5, 2, 0.2);
    
    std::vector<double> inputs = {0.1, 0.5, 0.9};
    std::vector<double> targets = {0.2, 0.8};
    
    // Train the original network
    for (int i = 0; i < 50; i++) {
        original.train(inputs, targets);
    }
    
    // Get output from original network
    auto original_output = original.query(inputs);
    
    // Serialize the network
    auto serialized_data = original.serializeToBytes();
    EXPECT_GT(serialized_data.size(), 0);
    
    // Create a new network and deserialize
    NeuralNetwork restored(3, 5, 2, 0.2);
    bool success = restored.deserializeFromBytes(serialized_data);
    EXPECT_TRUE(success);
    
    // Check that the restored network produces the same output
    auto restored_output = restored.query(inputs);
    EXPECT_EQ(restored_output.size(), original_output.size());
    
    for (size_t i = 0; i < original_output.size(); i++) {
        EXPECT_NEAR(restored_output[i], original_output[i], 1e-10);
    }
}

// Main function for running all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
