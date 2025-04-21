import gzip
import io
import pickle
import numpy as np

PATH = "e:\\Studia_Teleinformatyka_2022_2023\\VI_semestr\\UM\\MNIST-nn\\mnist.pkl.gz"


def read_data(path: str):
    with open(path, "rb") as file_gz:
        b_file_gz = file_gz.read()

    pkl_file = gzip.GzipFile(mode="rb", fileobj=io.BytesIO(b_file_gz))

    return pickle.load(pkl_file, encoding="latin1")

    # print(training_data[0].shape, training_data[1].size)
    # print(validation_data[0].shape, validation_data[1].size)

def one_hot_encoding(digits: np.ndarray):
    
    encoded_digits_matrix = np.zeros(shape=(digits.size, 10))
    encoded_digits_matrix[np.arange(digits.size), digits] = 1

    return encoded_digits_matrix

def transform_data():
    training_data, validation_data, test_data = read_data(PATH)
    inputs = training_data[0]
    labels = one_hot_encoding(training_data[1])
    
    # modifying training data format
    training_data = [(training_sample[0].reshape(training_sample[0].size, 1), training_sample[1].reshape(training_sample[1].size, 1)) for training_sample in zip(inputs, labels)]

    # modifying validation data format
    validation_data = [(validation_data[0][i].reshape(validation_data[0].shape[1], 1), validation_data[1][i]) for i in range(len(validation_data[0]))]
        
    # modifying test data
    test_data = [(test_data[0][i].reshape(test_data[0].shape[1], 1), test_data[1][i]) for i in range(len(test_data[0]))]

    return (training_data, validation_data, test_data)

