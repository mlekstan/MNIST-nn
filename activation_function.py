import numpy as np

def sigmoid(input: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-input))

def sigmoid_derivative(input: np.ndarray) -> np.ndarray:
    return sigmoid(input)*(1-sigmoid(input))