import numpy as np
import pickle
import time
from data import clean, get_data, clean_and_read

from activation_functions import *

EPOCHS = 100000
TRAINING_SIZE = 0.6
TESTING_SIZE = 0.4

np.random.seed(1)


def split_data():
    data = clean_and_read()
    np.random.shuffle(data)

    training_length = int(len(data) * TRAINING_SIZE)

    train = data[:training_length]
    test = data[training_length:]

    return train[:, 1:6], train[:, -1:], test[:, 1:6], test[:, -1:]


def generate_weights(n, shape):
    return np.random.normal(-2/n, 2/n, shape)


x, y, test_x, test_y = split_data()

syn0 = generate_weights(5, (5, 8))
syn1 = generate_weights(8, (8, 1))

bias0 = np.random.normal(0, 8)
bias1 = np.random.normal(0, 1)

syn = [syn0, syn1]
biases = [bias0, bias1]
activations = [Sigmoid, Linear]

for j in range(EPOCHS):
    error_list = []
    for s_input, s_output in zip(x, y):
        k0 = np.array([s_input])
        layers = [k0]

        for k in range(len(syn)):
            activation = activations[k]
            layers.append(activation.apply(np.dot(layers[-1], syn[k]) + biases[k]))

        output_activation = activations[-1]
        network_output = layers[-1]
        output_error = np.array([s_output]) - network_output
        output_delta = output_error * output_activation.deriv(network_output)

        error_list.append(output_error)
        errors = [output_error]
        deltas = [output_delta]
        hidden_layers = layers[1:-1]
        hidden_syn = syn[1:]

        # Only run on hidden layers
        for i, layer in enumerate(hidden_layers):
            error = deltas[-1].dot(hidden_syn[i].T)
            errors.append(error)
            activation = activations[i]
            delta = error * activation.deriv(layer)
            deltas.append(delta)

        for i, d in enumerate(reversed(deltas)):
            syn[i] += layers[i].T.dot(d) * 0.1
            biases[i] += d * 0.1

    if j % 1000 == 0:
        print('Error: {}'.format(np.mean(np.abs(error_list))))

errors = []
for x_, y_ in zip(test_x, test_y):
    k0 = np.array([x_])
    k1 = Sigmoid.apply(np.dot(k0, syn0) + bias0)
    k2 = Sigmoid.apply(np.dot(k1, syn1) + bias1)

    error = np.array([y_]) - k2
    errors.append(error)

print('Error: {}'.format(np.mean(np.abs(errors))))
