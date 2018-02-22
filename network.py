import numpy as np
import pandas as pd
from data import clean, get_data, clean_and_read

np.random.seed(1)


class Sigmoid:

    @staticmethod
    def apply(x):
        return np.array(1 / (1 + np.exp(-x)))

    @staticmethod
    def deriv(x):
        return np.array(x * (1 - x))


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
        return np.array(x)

    @staticmethod
    def deriv(x):
        return np.array([1])


class Adaline:
    @staticmethod
    def calculate(expected, prediction):
        return expected - prediction

    @staticmethod
    def propagate(x, y):
        return x.dot(y)

    @staticmethod
    def update_weights(weights, layer_output, delta, learning_rate):
        weights += layer_output.T.dot(delta * learning_rate)
        return weights


class GradientDescent:
    @staticmethod
    def calculate(expected, prediction):
        return np.square(expected - prediction)

    @staticmethod
    def propagate(x, y):
        return x.dot(y)

    @staticmethod
    def update_weights(weights, predicted, expected, layer_input, learning_rate, activation):
        pass
        # cost_gradient = 2 * (predicted - expected)  # derivative of cost function
        # print(cost_gradient.shape)
        # activation_gradient = activation.deriv(predicted)
        # # print(activation_gradient.shape)
        # weight_gradient = [[d] for d in layer_input]
        # # print(weight_gradient.shape)
        #
        # return weights - (cost_gradient * activation_gradient * weight_gradient * learning_rate)


class Layer(object):
    def __init__(self, inputs, units, activation, name=''):
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.name = name

        # self.weights = 2 * np.random.random((self.inputs, self.units)) - 1
        self.weights = np.random.normal(0, 0.01, (self.inputs, self.units))
        self.biases = np.random.normal(0, self.units)

    def activate(self, input_values):
        # if len(input_values[1]) != self.inputs:
        #     raise Exception('Input not the correct length. Given {} expected {}'.format(len(input_values), self.inputs))

        return self.activation.apply(np.dot(input_values, self.weights) + self.biases)

    def update(self, weight_delta, bias_delta):
        self.weights -= weight_delta
        self.biases -= bias_delta


class Network(object):
    def __init__(self, layers, learning_rate=1e-3):
        self.layers = layers
        self.learning_rate = learning_rate
        self.inputs = []
        self.output = None

    def run(self, data):
        self.inputs = []
        self.inputs.append(data)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                self.inputs.append(layer.activate(self.inputs[-1]))
            else:
                self.output = layer.activate(self.inputs[-1])

        return self.output

    def backpropagate(self, expected):

        layer_inputs = list(zip(self.layers, self.inputs))

        # Deal with output
        prediction = self.output
        error = prediction - expected
        d_error = np.array(2 * error)

        d_activation_o = self.layers[-1].activation.deriv(prediction)
        d_chain = d_error * d_activation_o * self.inputs[-1]

        weight_delta = np.array([[d] for d in d_chain])
        bias_delta = d_error * d_activation_o

        self.layers[-1].update(weight_delta * self.learning_rate, bias_delta * self.learning_rate)

        for l in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[l-1]
            d_chain *= layer.activation.deriv(self.inputs[l])

            weight_delta = []
            for i in range(layer.inputs):
                tmp = []
                for d_a in d_chain:
                    tmp.append(d_a * self.inputs[l - 1][i])
                weight_delta.append(tmp)

            weight_delta = np.array(weight_delta)
            bias_delta = d_chain
            layer.update(weight_delta * self.learning_rate, bias_delta * self.learning_rate)
        return error


INPUT_NODES = 5
HIDDEN_NODES = 2
OUTPUT_NODES = 1
LEARNING_RATE = 1e-1
TRAINING_PERCENTAGE = 0.7


def main():
    df = clean_and_read()

    values = df.values
    np.random.shuffle(values)

    training_length = int(len(values) * TRAINING_PERCENTAGE)
    train = values[:training_length]
    test = values[training_length:]

    x_train = train[:, :-1]
    y_train = train[:, -1:]
    x_test = test[:, :-1]
    y_test = test[:, -1:]

    layer_i = Layer(inputs=INPUT_NODES, units=HIDDEN_NODES, activation=Sigmoid, name='input')
    # layer_h = Layer(inputs=HIDDEN_NODES, units=HIDDEN_NODES, activation=Sigmoid, name='hidden')
    layer_o = Layer(inputs=HIDDEN_NODES, units=OUTPUT_NODES, activation=Linear, name='output')

    network = Network(layers=[layer_i, layer_o], learning_rate=LEARNING_RATE)

    for i in range(100000):
        errors = []
        # for x, y in zip(x_train, y_train):
        _ = network.run(x_train)
        errors += [network.backpropagate(y_train)]
        if i % 1000 == 0:
            error = np.mean(np.abs(errors))
            print('Average Error: {}'.format(error))

    predicted = network.run(x_test)
    error = y_test - predicted

    print('Error: {}'.format(np.mean(np.abs(error))))

    p = network.run(x_test[-1])
    error = y_test[-1] - p
    print('Expected: {} | Predicted: {} | Error: {}%'.format(y_test[-1], p, int(100*(np.abs(error) / y_test[-1]))))


if __name__ == '__main__':
    main()
