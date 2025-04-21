from nn_impl import Network
import data_loader

nn = Network([784, 30, 10])

training_data, validation_data, test_data = data_loader.transform_data()

nn.gradient_descent(training_data, 10, 3, 30, test_data)