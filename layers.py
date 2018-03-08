import numpy as np
from activation_functions import *
from data import clean, get_data, clean_and_read
np.random.seed(1)

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
        # self.weights = np.random.normal(-2/self.inputs, 2/self.inputs, (self.inputs, self.outputs))
        self.weights = np.random.normal(0, 0.01, (self.inputs, self.outputs))

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
        for l in self.layers:
            input_value = l.activate(input_value)
        return np.array(input_value)

    def optimise(self, prediction, target):
        # prediction is the output of the final layer in the network
        error = 0.5 * np.sum((target - prediction) ** 2)  # change this to plug and play class
        error_deriv = (target - prediction)

        output_layer = self.layers[-1]
        output_delta = error_deriv * output_layer.deriv()

        deltas = [output_delta]

        for l in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[l]
            errors = deltas[-1].dot(self.layers[l + 1].weights.T)
            deltas.append(errors * layer.deriv())

        deltas = list(reversed(deltas))
        for l, layer in enumerate(self.layers[1:]):
            weight_change = layer.values.T.dot(deltas[l]) * self.learning_rate
            layer.weights += weight_change
        return error_deriv


def split_data():
    data = clean_and_read()
    np.random.shuffle(data)

    training_length = int(len(data) * 0.6)

    train = data[:training_length]
    test = data[training_length:]

    return train[:, :5], train[:, -1:], test[:, :5], test[:, -1:]


x, y, test_x, test_y = split_data()

l_i = InputLayer(6, 6)
l_h = HiddenLayer(6, 8, Sigmoid)
l_o = OutputLayer(8, 1, Linear)

n = Network(layers=[l_i, l_h,  l_o], learning_rate=1e-3)

x = x.tolist()
# bias
for j in range(len(x)):
    x[j] += [1]
x = np.array(x)

test_x = test_x.tolist()
# bias
for j in range(len(test_x)):
    test_x[j] += [1]
test_x = np.array(test_x)

for i in range(100000):
    errors = []
    prediction = n.run(x)
    error = n.optimise(prediction, np.array(y))
    errors.append(error)
    if i % 10000 == 0:
        print(np.mean(np.abs(errors)))

for _x, _y in zip(test_x[0:10], test_y[0:10]):
    prediction = n.run(np.array([_x]))
    # error = n.optimise(prediction, np.array([_y]))
    print('Prediction: {} | Actual: {}'.format(prediction, [_y]))
