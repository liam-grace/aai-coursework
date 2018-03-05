import numpy as np
import pickle
import time
from data import clean, get_data, clean_and_read
from visualise import draw_graph

from activation_functions import *

INPUTS = 5
OUTPUTS = 1

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

    return train[:, :-1], train[:, -1:], test[:, :-1], test[:, -1:]


def generate_weights(n, shape):
    return np.random.normal(-2/n, 2/n, shape)


x, y, test_x, test_y = split_data()

syn0 = generate_weights(6, (6, 8))
syn1 = generate_weights(8, (8, 1))

syn = [syn0, syn1]
count = 0
for j in range(EPOCHS):
    error_list = []
    for s_input, s_output in zip(x, y):
        # print(s_input, s_output)
        k0 = np.array([s_input])
        k1 = Sigmoid.apply(np.dot(k0, syn0))
        k2 = Sigmoid.apply(np.dot(k1, syn1))

        layers = [k0, k1, k2]

        k2_error = np.array([s_output]) - k2
        k2_delta = k2_error * Sigmoid.deriv(k2)
        error_list.append(k2_error)
        errors = [k2_error]
        deltas = [k2_delta]
        hidden_layers = layers[1:-1]
        hidden_syn = syn[1:]

        # Only run on hidden layers
        for i, layer in enumerate(hidden_layers):
            error = deltas[-1].dot(hidden_syn[i].T)
            errors.append(error)
            delta = error * Sigmoid.deriv(layer)
            deltas.append(delta)

        for i, d in enumerate(reversed(deltas)):
            # print(syn[i].shape, layers[i].T.shape, d.shape)
            syn[i] += layers[i].T.dot(d) * 0.1
        count += 1
    if j % 10000 == 0:
        print('Error: {}'.format(np.mean(np.abs(error_list))))

with open('weights', 'wb') as f:
    pickle.dump(syn, f)

errors = []
for x_, y_ in zip(test_x, test_y):
    k0 = np.array([x_])
    k1 = Sigmoid.apply(np.dot(k0, syn0))
    k2 = Sigmoid.apply(np.dot(k1, syn1))

    error = np.array([y_]) - k2
    errors.append(error)

print('Error: {}'.format(np.mean(np.abs(errors))))
