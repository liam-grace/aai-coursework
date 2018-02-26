import numpy as np


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
        return 1
