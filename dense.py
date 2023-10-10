import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.wieghts =  np.random.rand(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)


    def forward(self, input):
        self.input = input
        return np.dot(self.wieghts, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weighted_gradient = np.dot(output_gradient, self.input.T)
        self.wieghts -= learning_rate * weighted_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.wieghts.T, output_gradient)

    def get_params(self):
        return {'weights': self.wieghts, 'bias': self.bias}

    def set_params(self, params):
        self.wieghts = params['weights']
        self.bias = params['bias']