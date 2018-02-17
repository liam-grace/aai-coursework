import random
import numpy as np


# [0] Bias
# [1:2] Data
# [3] Expected result
sample_data = [
    [1, 1, 4, 1],
    [1, 2, 9, 1],
    [1, 5, 6, 1],
    [1, 4, 5, 1],
    [1, 6, 0.7, -1],
    [1, 1, 1.5, -1]
]


def default_activation(x):
    return x


def zero_activation(x):
    return -1 if x < 0 else 1


class Perceptron(object):
    def __init__(self, num_inputs, activation=default_activation):
        self.activation = activation
        self.num_inputs = num_inputs
        self.weights = np.random.rand(num_inputs)
        self.value = 0

    def activate(self, input_value):
        value = sum([i * w for i, w in zip(input_value, self.weights)])
        self.value = value
        return self.value

    def learn(self, input_value):
        expected = input_value[-1]
        for i, (ip, w) in enumerate(zip(input_value[0:-1], self.weights)):
            self.weights[i] += expected * ip


class Layer(object):
    def __init__(self, num_inputs, num_nodes, activation=default_activation):
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.activation = activation
        self.nodes = []
        self.build_layer()

    def build_layer(self):
        for _ in range(self.num_nodes):
            node = Perceptron(self.num_inputs, self.activation)
            self.nodes.append(node)

    def activate(self, input_value):
        values = [n.activate(input_value) for n in self.nodes]
        return values

    def learn(self, input_value):
        for n in self.nodes:
            n.learn(input_value)


if __name__ == '__main__':
    pass


