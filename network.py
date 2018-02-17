import numpy as np


def tanh(val):
    return np.tanh(val)


def inverse_tanh(val):
    return 1 - np.tanh(val) ** 2


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def inverse_sigmoid(val):
    return val * (1 - val)


def calc_delta(error, layer, inverse):
    return error * inverse(layer)


def calc_error(delta, weights):
    return delta.dot(weights.T)


def next_layer(this_layer, weights, activation):
    return activation(this_layer.dot(weights))


def new_weights(weights, layer, next_layer_delta):
    return weights + layer.T.dot(next_layer_delta) * LEARNING_RATE


INPUT_NODES = 3
HIDDEN_NODES = 4
OUTPUT_NODES = 1
LEARNING_RATE = 0.01

x = np.array([
    [1, 1, 4],
    [1, 2, 9],
    [1, 5, 6],
    [1, 4, 5],
    [1, 6, 0.7],
    [1, 1, 1.5]
])

y = np.array([
    [1],
    [1],
    [1],
    [1],
    [0],
    [0]
])


w1 = np.random.random((INPUT_NODES, HIDDEN_NODES))  # (3,4) matrix. 3 inputs, 4 hidden nodes
w2 = np.random.random((HIDDEN_NODES, OUTPUT_NODES))  # (4,1) matrix. 4 hidden nodes, 1 output

c = 0
for _ in range(1000000):
    done = False
    for j, i in enumerate(x):
        l0 = np.array([i])  # (3,) matrix. 3 inputs

        l1 = next_layer(l0, w1, sigmoid)  # (4,) matrix. 4 nodes

        l2 = next_layer(l1, w2, sigmoid)  # (1,) matrix. 1 node

        l2_error = y[j] - l2

        if c % 100000 == 0:
            print('Error: {}'.format(np.abs(np.mean(l2_error))))

        l2_delta = calc_delta(l2_error, l2, inverse_sigmoid)

        l1_error = calc_error(l2_delta, w2)
        l1_delta = calc_delta(l1_error, l1, inverse_sigmoid)

        w2 = new_weights(w2, l1, l2_delta)
        w1 = new_weights(w1, l0, l1_delta)
        c += 1

        if np.abs(np.mean(l2_error)) < 0.005:
            done = True
            print(np.abs(np.mean(l2_error)))
            break

    if done:
        break

for i in range(len(x)):
    l0 = np.array(x[i])
    l1 = next_layer(l0, w1, sigmoid)
    l2 = next_layer(l1, w2, sigmoid)

    print(l2, y[i])


