import numpy as np
import pandas as pd
from data import clean, get_data, clean_and_read

np.random.seed(1)


class Sigmoid:

    @staticmethod
    def apply(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def deriv(x):
        return x * (1 - x)


class TanH:
    @staticmethod
    def apply(x):
        return np.tanh(x)

    @staticmethod
    def deriv(x):
        return 1 - np.tanh(x) ** 2


class Linear:
    @staticmethod
    def apply(x):
        return x

    @staticmethod
    def deriv(x):
        return 1


class Adaline:
    @staticmethod
    def calculate_output_layer(expected, prediction):
        return expected - prediction

    def calculate_hidden_layer(self):
        return


class Layer(object):
    def __init__(self, inputs, units, activation, name=''):
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.name = name

        # self.weights = 2 * np.random.random((self.inputs, self.units)) - 1
        self.weights = np.random.normal(0, 0.01, (self.inputs, self.units))

    def activate(self, input_values):
        if len(input_values[1]) != self.inputs:
            raise Exception('Input not the correct length. Given {} expected {}'.format(len(input_values), self.inputs))

        return self.activation.apply(input_values.dot(self.weights))


class Network(object):
    def __init__(self, layers, learning_rate=1e-3):
        self.layers = layers
        self.learning_rate = learning_rate

        self.layer_data = []

    def run(self, data):
        self.layer_data = []
        self.layer_data.append(data)
        for layer in self.layers:
            self.layer_data.append(layer.activate(self.layer_data[-1]))

        return self.layer_data[-1]

    def backpropagate(self, expected):
        error = expected - self.layer_data[-1]
        delta = error * self.layers[-1].activation.deriv(self.layer_data[-1])
        deltas = [delta]
        for l in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[l]
            errors = deltas[-1].dot(layer.weights.T)
            deltas.append(errors * layer.activation.deriv(self.layer_data[l]))

        for i, layer in enumerate(self.layers):
            layer.weights += self.layer_data[i].T.dot(deltas[-(i + 1)] * self.learning_rate)


INPUT_NODES = 5
HIDDEN_NODES = 8
OUTPUT_NODES = 1
LEARNING_RATE = 1e-4
TRAINING_PERCENTAGE = 0.6


def main():
    df = clean_and_read()

    values = df.values
    np.random.shuffle(values)

    train = values[:int(len(values) * TRAINING_PERCENTAGE)]
    test = values[int(len(values) * TRAINING_PERCENTAGE):]

    x_test = test[:, :-1]
    y_test = test[:, -1:]
    x_train = train[:, :-1]
    y_train = train[:, -1:]

    layer_i = Layer(inputs=len(x_train[1]), units=8, activation=Sigmoid, name='input')
    layer_h = Layer(inputs=8, units=8, activation=Sigmoid, name='hidden')
    layer_o = Layer(inputs=8, units=1, activation=Linear, name='output')

    network = Network(layers=[layer_i, layer_h, layer_o], learning_rate=LEARNING_RATE)

    for i in range(5000000):
        prediction = network.run(x_train)
        network.backpropagate(y_train)
        if i % 10000 == 1:
            error = np.mean(np.abs(y_train - prediction))
            print('Average Error: {}'.format(error))

    predicted = network.run(x_test)
    error = y_test - predicted

    # for actual, prediction in zip(y_test, predicted):
    #     print('Expected: {} | Prediction: {} | Error: {}'.format(actual, prediction, np.mean(np.abs(actual-prediction))))

    print('Error: {}'.format(np.mean(np.abs(error))))


if __name__ == '__main__':
    main()

