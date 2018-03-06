import numpy as np
from activation_functions import *
from data import clean, get_data, clean_and_read


class Layer:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.values = None
        self.activation_values = None

    def activate(self, input_value):
        raise Exception('Activate not implemented')


class InputLayer(Layer):
    def activate(self, input_value):
        self.values = input_value
        self.activation_values = input_value
        return input_value


class HiddenLayer(InputLayer):
    def __init__(self, inputs, outputs, activation_function=None):
        super(HiddenLayer, self).__init__(inputs, outputs)
        if activation_function is None:
            activation_function = Sigmoid
        self.activation_values = None
        self.activation_function = activation_function
        self.weights = np.random.normal(-2/self.inputs, 2/self.inputs, (self.inputs, self.outputs))
        self.biases = np.random.normal(-2/self.inputs, 2/self.inputs, self.outputs)

    def activate(self, input_value):
        self.values = input_value
        self.activation_values = self.activation_function.apply(np.dot(input_value, self.weights))

        return self.activation_values

    def deriv(self):
        return self.activation_function.deriv(self.activation_values)


class OutputLayer(HiddenLayer):
    pass


class Network:
    def __init__(self, layers=None, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate

    def run(self, input_value):
        input_value = np.array(input_value)
        for l in self.layers:
            input_value = l.activate(input_value)
        return input_value

    def optimise(self, prediction, target):
        error = (target - prediction)
        deltas = []
        b_deltas = []

        output_layer = self.layers[-1]
        delta_output = error * output_layer.deriv()
        deltas.append(delta_output)
        b_deltas.append(np.sum(delta_output, keepdims=True))

        hidden_layers = self.layers[1:-1]

        for i, l in enumerate(reversed(hidden_layers)):
            layer_deriv = l.deriv()
            next_layer_weights = self.layers[i + 2].weights
            delta = deltas[-1]
            delta = delta.dot(next_layer_weights.T) * layer_deriv
            deltas.append(delta)
            b_deltas.append(np.sum(delta, keepdims=True))

        self.layers[2].weights += self.layers[1].activation_values.T.dot(deltas[0]) * self.learning_rate
        self.layers[1].weights += self.layers[0].activation_values.T.dot(deltas[1]) * self.learning_rate



        return error


def split_data():
    data = clean_and_read()
    np.random.shuffle(data)

    training_length = int(len(data) * 0.6)

    train = data[:training_length]
    test = data[training_length:]

    return train[:, 1:6], train[:, -1:], test[:, 1:6], test[:, -1:]


x, y, test_x, test_y = split_data()

l_i = InputLayer(5, 5)
l_h = HiddenLayer(5, 8, Sigmoid)
l_o = OutputLayer(8, 1, Sigmoid)

n = Network(layers=[l_i, l_h, l_o], learning_rate=0.1)

# for _ in range(10000000):
    # for x_, y_ in zip(x, y):
    # prediction = n.run(np.array([[2, 2]]))
    # print(prediction)
    # error = n.optimise(prediction, np.array([2, 2]))
    # print(np.abs(np.mean(error)))

for _ in range(10000000):
    errors = []
    # for _x, _y in zip(x, y):
    prediction = n.run(np.array(x))
    error = n.optimise(prediction, np.array(y))
    errors.append(error)
    print(np.mean(np.abs(error)))
