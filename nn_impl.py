from typing import List, Tuple
import random
import numpy as np
from activation_function import sigmoid, sigmoid_derivative


class Network:
    def __init__(self, layers_sizes: List) -> None:
        self.layers_sizes = layers_sizes
        self.layers_weights = [np.random.randn(layers_sizes[i], layers_sizes[i-1]) for i in range(1, len(layers_sizes))]
        self.layers_biases = [np.random.randn(layers_sizes[i], 1) for i in range(1, len(layers_sizes))]


    def pass_forward(self, layer_input: np.ndarray) -> Tuple:
        weighted_inputs = []
        outputs = [layer_input]
        
        for layer_weights, layer_biases in zip(self.layers_weights, self.layers_biases):
            
            weighted_layer_input = layer_weights @ layer_input + layer_biases
            weighted_inputs.append(weighted_layer_input)
            
            layer_output = sigmoid(weighted_layer_input)
            outputs.append(layer_output)
            
            layer_input = layer_output

        return (weighted_inputs, outputs)
    

    def backpropagation(self, network_input: np.ndarray, label: np.ndarray) -> Tuple:
        weighted_inputs, outputs = self.pass_forward(network_input)
        layers_errors = []
        grad_w = []
        
        for i in range(1, len(weighted_inputs)+1):
            if i == 1:
                layer_errors = (outputs[-i] - label) * sigmoid_derivative(weighted_inputs[-i])
                
            else:
                layer_errors = (self.layers_weights[-i+1].transpose() @ layer_errors) * sigmoid_derivative(weighted_inputs[-i])

            layers_errors.append(layer_errors)
            grad_w.append(layer_errors @ outputs[-i-1].transpose())
        
        grad_w = grad_w[::-1]
        grad_b = layers_errors[::-1]

        return (grad_w, grad_b)


    def select_mini_batches(self, training_data: List, mini_batch_size: int) -> List:
        random.shuffle(training_data)
        mini_batches = [training_data[i:mini_batch_size+i] for i in range(0, len(training_data), mini_batch_size)]

        return mini_batches


    def update_net_params(self, mini_batch: List, learning_rate: float):

        avg_grad_w = [np.zeros(layer_weights.shape) for layer_weights in self.layers_weights]
        avg_grad_b = [np.zeros(layer_biases.shape) for layer_biases in self.layers_biases]

        for input, label in mini_batch:
            grad_w, grad_b = self.backpropagation(input, label)
            
            avg_grad_w = [layer_avg_grad_w + (layer_grad_w/len(mini_batch)) for layer_avg_grad_w, layer_grad_w in zip(avg_grad_w, grad_w)]
            avg_grad_b = [layer_avg_grad_b + (layer_grad_b/len(mini_batch)) for layer_avg_grad_b, layer_grad_b in zip(avg_grad_b, grad_b)]

        self.layers_weights = [layer_weights - learning_rate * layer_avg_grad_w for layer_weights, layer_avg_grad_w in zip(self.layers_weights, avg_grad_w)]
        self.layers_biases = [layer_biases - learning_rate * layer_avg_grad_b for layer_biases, layer_avg_grad_b in zip(self.layers_biases, avg_grad_b)]


    def gradient_descent(self, training_data: List, mini_batch_size: int, learning_rate: float, epochs: int, test_data: List):
        for epoch in range(epochs):    
            mini_batches = self.select_mini_batches(training_data, mini_batch_size)
            for mini_batch in mini_batches:
                self.update_net_params(mini_batch, learning_rate)

            errors_count = self.check_correctness(test_data)
            print(f"Epoch {epoch}, {errors_count}/{len(test_data)} ")


    def check_correctness(self, test_data: List):
        errors_count = 0
        for test_input, test_label in test_data:
            _, outputs = self.pass_forward(test_input)
            net_output = np.argmax(outputs[-1])

            if test_label != net_output:
                errors_count += 1

        return len(test_data) - errors_count
