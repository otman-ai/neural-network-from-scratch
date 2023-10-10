import numpy as np
from dense import Dense
from activations import Tanh
from loss import mse, mse_prime
from network import train
import pickle
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
network = [
    Dense(2,3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
train(network, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.1)
