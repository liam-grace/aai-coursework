import numpy as np


class Sigmoid:
    @staticmethod
    def apply(x):
        return np.array(1 / (1 + np.exp(-x)))

    @staticmethod
    def deriv(x):
        return np.array(x * (1 - x))

    def __repr__(self):
        return 'Sigmoid'


class TanH:
    @staticmethod
    def apply(x):
        return np.tanh(x)

    @staticmethod
    def deriv(x):
        return 1 - np.tanh(x) ** 2

    def __repr__(self):
        return 'TanH'


class Linear:
    @staticmethod
    def apply(x):
        return np.array(x)

    @staticmethod
    def deriv(x):
        return np.array(1)

    def __repr__(self):
        return 'Linear'


class Relu:
    @staticmethod
    def apply(x):
        return np.maximum(np.zeros_like(x), x)

    @staticmethod
    def deriv(x):
        x[x < 0] = 0
        return x


class LeakyRelu:
    @staticmethod
    def apply(x):
        def get_map(y):
            return 1 if y > 0 else 0.01 * y

        v = np.vectorize(get_map)

        return v(x)

    @staticmethod
    def deriv(x):
        x[x < 0] = 0.01
        return x
