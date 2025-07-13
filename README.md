# Nermal Neural Network

Hello, this project is my start to understanding neural networks. I named my neural network Nermal because I like Garfield, and it kind of sounds like neural. I started with a book from Tariq Rashid. His Github repository can be found here [makeyourownneuralnetwork](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork). And his book is Make Your Own Neural Network. I enjoyed it and would recommend.

I went through his examples and developed a neural network in Python. But then converted the project to C++ using Eigen. And then implemented a small UI to help demonstrate how the neural network is trained and how it makes predictions.

Training data comes from [mnist](https://github.com/phoebetronic/mnist).

I used copilot / Claude Sonnet to help convert from python and to help build the UI. 

It was a fun little project.


Some additional comments generated with the help of Copilot.

## How the Neural Network Works

This neural network is designed to recognize handwritten digits (0-9) from 28x28 pixel images. Here's how it operates:

### Network Architecture
- **784 input nodes** (28×28 pixels flattened into a single vector)
- **100 hidden nodes** (the "feature detectors")
- **10 output nodes** (one for each digit 0-9)

### Weight Matrices: The Brain's Connections
The network uses two weight matrices to connect the layers:
- **Input→Hidden weights** (100×784): Each element represents the connection strength from a specific pixel to a specific hidden node
- **Hidden→Output weights** (10×100): Each element represents how much each hidden node influences each digit prediction

Think of weights as the "strength of belief" - positive weights excite neurons, negative weights inhibit them. During training, these weights adjust to recognize patterns.

### The Sigmoid Function: Making Decisions
The network uses the sigmoid activation function: σ(x) = 1/(1 + e^(-x))
- Maps any input to a value between 0 and 1
- Acts like a "soft switch" - gradually turns neurons on/off rather than hard binary decisions
- Has a useful mathematical property: its derivative is σ(x)(1-σ(x)), which simplifies learning calculations

### Processing the Very First Training Image
When the network sees its first handwritten "3":

1. **Random Start**: All 784×100 + 100×10 = 79,400 weights start as small random numbers
2. **Forward Pass**: 
   - Each pixel value (0.01 to 0.99) gets multiplied by random weights
   - Hidden nodes receive weighted sums of ALL pixels
   - Since weights are random, hidden nodes produce essentially random outputs
   - Output layer makes a random guess (maybe predicting "7" when it should predict "3")
3. **Error Calculation**: Network compares its random guess to the correct answer
4. **Backpropagation**: The learning algorithm adjusts weights based on the error:
   - If a hidden node contributed to a wrong answer, its connections get weakened
   - If a hidden node could have helped the right answer, its connections get strengthened

### After Thousands of Training Images
The magic happens through repetition:

**Hidden Node Specialization**: The 100 hidden nodes develop specialized roles:
- Some detect curves in specific image regions
- Others recognize vertical lines, horizontal segments, or diagonal strokes
- Many develop overlapping specializations for robustness

**Pattern Recognition**: For a "3", the trained network might activate:
- Hidden nodes that detect right-facing curves (strongly active)
- Hidden nodes that detect horizontal middle segments (moderately active)  
- Hidden nodes that detect closed loops (inactive - distinguishes from "8")

**Output Layer Learning**: Each output node learns which combination of hidden node activations represents its digit:
- Output node "3" learns to fire when the "right curves + middle horizontal" pattern appears
- Output node "8" learns to fire when the "closed loops + curves" pattern appears

### Learning Rate: Controlling the Pace
The learning rate (0.1 in this implementation) controls how quickly weights change:
- Too high (e.g., 5.0): Network makes huge adjustments and becomes unstable
- Too low (e.g., 0.001): Network learns too slowly, may never reach good performance
- Just right (0.1): Steady, stable progress toward accurate digit recognition

### The End Result
After training on thousands of diverse handwritten digits, the network develops an internal representation of what makes each digit unique. When shown a new handwritten digit, it can recognize it by comparing the activation patterns of its hidden nodes to the learned patterns for each digit class.

