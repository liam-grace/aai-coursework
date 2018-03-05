import numpy as np
import matplotlib
from data import clean, get_data, clean_and_read
from visualise import draw_graph

from activation_functions import *

INPUTS = 5
OUTPUTS = 1

EPOCHS = 100000
TRAINING_SIZE = 0.5
TESTING_SIZE = 0.25

np.random.seed(1)


def generate_random_networks(n):
    hidden_nodes = [1, 2, 3, 5, 8, 13]
    activation_functions = [Linear, Sigmoid]
    learning_rates = [1e-3, 1e-2, 1e-1, 1]
    batch_sizes = [2, 4, 8, 16, 32, 64]

    networks = []
    for _ in range(n):
        network = Network([
            (INPUTS, None),
            (np.random.choice(hidden_nodes), np.random.choice(activation_functions)),
            (OUTPUTS, np.random.choice(activation_functions))],
            learning_rate=np.random.choice(learning_rates))

        networks.append((network, np.random.choice(batch_sizes)))
    return networks


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

        self.outputs = []
        self.synapses = []
        self.biases = []

        self.activations = []

        for s in structure[1:]:
            self.activations.append(s[1])
        for i, (s, activation) in enumerate(structure[:-1]):
            self.synapses.append(np.random.normal(0, 0.01, (s, structure[i+1][0])))
            self.biases.append(np.random.normal(0, size=structure[i+1][0]))

    def run(self, input_value):
        self.outputs = [input_value]

        for i, synapse in enumerate(self.synapses):
            activation = self.activations[i]
            self.outputs.append(activation.apply(np.dot(self.outputs[-1], synapse)))

        return self.outputs[-1]

    def optimise(self, expected):
        # mean = np.square(np.subtract(expected, self.outputs[-1])).mean()
        # error = np.full_like(expected, mean)
        # mean_error = np.mean(np.abs(error))

        error = expected - self.outputs[-1]

        lo_delta = error * self.activations[-1].deriv(self.outputs[-1])
        deltas = [lo_delta]

        for l in range(len(self.synapses) - 1, 0, -1):
            error = deltas[-1].dot(self.synapses[l].T)
            activation = self.activations[l]
            delta = error * activation.deriv(self.outputs[l])
            deltas.append(delta)

        for i, d in enumerate(deltas):
            output = self.outputs[-(i-1)]
            change = output.T.dot(d)
            self.synapses[-(i-1)] += change * self.learning_rate
        return np.mean(np.abs(error))


def chunk(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def main():
    train_x, train_y, test_x, test_y = split_data()

    error_list = []

    network = Network([(5, None), (9, Sigmoid), (1, Linear)], learning_rate=0.01)
    for i in range(EPOCHS):
        errors = []
        for x, y in zip(train_x, train_y):
            x = np.resize(x, (1, INPUTS))
            y = np.resize(y, (1, OUTPUTS))
            prediction = network.run(np.array(x))
            network.optimise(np.array(y))
            errors += [prediction - np.array(y)]
        if i % 10000 == 0:
            error = np.mean(np.abs(errors))
            error_list.append(error)
            print('Error: {}'.format(np.mean(np.abs(errors))))

    draw_graph(error_list)

    test_x = test_x[:5]
    test_y = test_y[:5]

    for x, y in zip(test_x, test_y):
        prediction = network.run(np.array([x]))
        error = prediction - np.array([y])
        print('Expected: {} | Prediction: {} | Error: {}'.format(y, prediction, error))


if __name__ == '__main__':
    main()

