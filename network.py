import numpy as np
import pandas as pd
from data import clean, get_data, clean_and_read

from activation_functions import *

EPOCHS = 100000
TRAINING_SIZE = 0.7

np.random.seed(1)


def split_data():
    data = clean_and_read()
    np.random.shuffle(data)

    training_length = int(len(data) * TRAINING_SIZE)

    train = data[:training_length]
    test = data[training_length:]

    return train[:, :-1], train[:, -1:], test[:, :-1], test[:, -1:]


class Network:
    def __init__(self, structure, learning_rate=0.1):
        self.structure = structure
        self.learning_rate = learning_rate

        self.layers = []
        self.synapses = []
        self.biases = []

        self.activations = []

        for s in structure:
            self.activations.append(s[1])
        for i, (s, activation) in enumerate(structure[:-1]):
            self.synapses.append(np.random.normal(0, 0.01, (s, structure[i+1][0])))
            self.biases.append(np.random.normal(0, size=structure[i+1][0]))

    def run(self, input_value):
        self.layers = [input_value]

        for i, synapse in enumerate(self.synapses):
            activation = self.activations[i + 1]
            self.layers.append(activation.apply(np.dot(self.layers[-1], synapse) + self.biases[i]))

        return self.layers[-1]

    def optimise(self, expected):
        error = self.layers[-1] - expected
        deltas = []

        for i, layer in enumerate(reversed(self.layers[1:])):
            if len(deltas) > 0:
                error = deltas[-1].dot(self.synapses[-i].T)
            activation = self.activations[-(i+1)]
            deltas.append(error * activation.deriv(layer))
        for s in range(len(self.synapses)):
            # print(self.synapses[s].shape, (self.layers[s].T.dot(list(reversed(deltas))[s])).shape)
            self.synapses[s] -= self.learning_rate * (self.layers[s].T.dot(deltas[-(s - 1)]))
            # print(self.biases[s].shape, deltas[-(s-1)])
            self.biases[s] -= self.learning_rate * deltas[-(s-1)][0]


def main():
    train_x, train_y, test_x, test_y = split_data()
    # train_x = [[1, 2, 3, 4, 5]]
    # train_y = [[1]]
    network = Network([(5, None), (9, Sigmoid), (1, Linear)], learning_rate=0.01)
    for i in range(EPOCHS):
        errors = []
        for x, y in zip(train_x, train_y):
            prediction = network.run(np.array([x]))
            # print(prediction)
            network.optimise(np.array([y]))
            errors.append(prediction - np.array([y]))
        if i % 1000 == 0:
            print('Error: {}'.format(np.mean(np.abs(errors))))

    test_x = test_x[:5]
    test_y = test_y[:5]

    for x, y in zip(test_x, test_y):
        prediction = network.run(np.array([x]))
        error = prediction - np.array([y])
        print('Expected: {} | Prediction: {} | Error: {}'.format(y, prediction, error))


if __name__ == '__main__':
    main()

