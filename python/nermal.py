#!/usr/bin/env python3

import numpy
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend which works well with Wayland
import matplotlib.pyplot
import scipy.special

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        # wih = Weights Input-to-Hidden
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))

        # who = Weights Hidden-to-Output
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # lr = Learning Rate
        self.lr = learningRate

        self.activationFunction = lambda x: scipy.special.expit(x)  # Sigmoid function

        pass

    def train(self, inputsList, targetsList):
        inputs = numpy.array(inputsList, ndmin=2).T
        targets = numpy.array(targetsList, ndmin=2).T

        hiddenInputs = numpy.dot(self.wih, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = numpy.dot(self.who, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        outputErrors = targets - finalOutputs
        hiddenErrors = numpy.dot(self.who.T, outputErrors)

        self.who += self.lr * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))
        self.wih += self.lr * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))

        pass

    def query(self, inputLists):
        inputs = numpy.array(inputLists, ndmin=2).T

        hiddenInputs = numpy.dot(self.wih, inputs)

        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = numpy.dot(self.who, hiddenOutputs)

        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs

    def __str__(self):
        return f"""Neural Network:
  Input Nodes: {self.inodes}
  Hidden Nodes: {self.hnodes}
  Output Nodes: {self.onodes}
  Learning Rate: {self.lr}
  
  Input-to-Hidden Weights Shape: {self.wih.shape}
  Hidden-to-Output Weights Shape: {self.who.shape}
  
  Input-to-Hidden Weights:
{self.wih}
  
  Hidden-to-Output Weights:
{self.who}"""

# Test out the neural network. We will traing with mnist data. These consist of 28x28 pixel images stored in an .csv file.

trainingDataFile = open("mnist_train.csv", 'r')
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()

# 28x28 = 784 pixels and input nodes
inputNodes = 784

# 100 hidden nodes (hidden nodes start to detect patterns like edges, curves, etc.)
hiddenNodes = 100

# 10 output nodes (one for each digit 0-9)
outputNodes = 10

# influences how quickly the neural network learns. If the number is too high, the network may not converge. 
# If it is too low, it will take a long time to learn.
learningRate = 0.3

nermal = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

# read from the training data
for record in trainingDataList:
    allValues = record.split(',')
    inputs = (numpy.asarray(allValues[1:], dtype=float) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(outputNodes) + 0.01
    targets[int(allValues[0])] = 0.99
    #train on the record
    nermal.train(inputs, targets)
    pass

# test data also consists of 28x28 pixel images stored in an .csv file. These values are not part of the training data and are 
# instead used to test to see if the neural network has learned to recognize digits.
testDataFile = open("mnist_test.csv", 'r')
testDataList = testDataFile.readlines()
testDataFile.close()

allValues = testDataList[9].split(',')
imageArray = numpy.asarray(allValues[1:], dtype=float).reshape((28,28))
matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

# print the test results
print(f"\n=== TESTING ON DIGIT ===")
print(f"Actual digit label: {allValues[0]}")
print("Neural network output (confidence for each digit 0-9):")
result = nermal.query((numpy.asarray(allValues[1:], dtype=float) / 255.0 * 0.99) + 0.01)
for i, confidence in enumerate(result.flatten()):
    print(f"  Digit {i}: {confidence:.4f}")
print(f"Predicted digit: {numpy.argmax(result)}")
print(f"Confidence: {numpy.max(result):.4f}")