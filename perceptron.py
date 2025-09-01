import math
import random

class Perceptron:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Perceptron, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.weights = None
            self.bias = None
            self.initialized = True
    
    def initialize_weights_bias(self, input_size):
        # self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        # self.bias = random.uniform(-1, 1)
        # Initialize weights and bias to 0.5 to ensure consistency with spreadsheet
        self.weights = [0.5 for _ in range(input_size)]
        self.bias = 0.5
        return self.weights, self.bias
    
    def dot_product(self, inputs):
        result = 0
        for i in range(len(inputs)):
            result += inputs[i] * self.weights[i]
        return result + self.bias
    
    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1
    
    def error_function(self, predicted, actual):
        return actual - predicted
    
    def update_weights_bias(self, inputs, target, predicted, learning_rate=0.1):
        delta = 2 * (predicted - target) * (1 - predicted) * predicted
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * delta * inputs[i]

        # Update bias
        self.bias -= learning_rate * delta
        
        return self.weights, self.bias