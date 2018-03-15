import multiprocessing
from activation_functions import *
from data import clean, get_data, clean_and_read
import matplotlib.pyplot as plt

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

        self.last_weight_delta = 0

    def activate(self, input_value):
        #  set the layers input value
        self.values = input_value
        #  store the layers output value. The dot product of the input value and the layers' weights, with
        #  the activation function applied to the entire matrix
        self.activation_values = self.activation_function.apply(np.dot(input_value, self.weights))
        #  return the layers output value
        return self.activation_values

    def deriv(self):
        return self.activation_function.deriv(self.activation_values)

    def update_weights(self, delta, momentum=False):
        self.weights += delta
        if momentum:
            self.weights += self.last_weight_delta * 0.9
        self.last_weight_delta = delta


class OutputLayer(HiddenLayer):
    pass


class Network:
    def __init__(self, layers=None, learning_rate=0.1):
        self.layers = layers
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

    def run(self, input_value):
        #  for each layer in the network
        for l in self.layers:
            #  pass in the current layers input value and store the output value
            input_value = l.activate(input_value)
        #  return the last layer output value
        return np.array(input_value)

    def optimise(self, prediction, target, momentum=False, update_weights=True):
        # prediction is the output of the final layer in the network
        error = 0.5 * np.sum((target - prediction) ** 2)  # Mean Squared Error function
        error_deriv = (target - prediction)  # Differential of Mean Squared Error

        output_layer = self.layers[-1]  # Output layer is the last layer in the layers array
        # Output layer delta calculation (apply the derivative of the output layers activation function)
        output_delta = error_deriv * output_layer.deriv()
        deltas = [output_delta]  # Start a list of deltas

        #  Reversed for loop, starting from layer feeding the output layer (the final hidden layer)
        for l in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[l]
            # Dot product of the last delta to be calculated and the weights. Note: A layers weights are to the left
            errors = deltas[-1].dot(self.layers[l + 1].weights.T)
            deltas.append(errors * layer.deriv())  # Calculate the delta

        if update_weights:
            deltas = list(reversed(deltas))  # reversing deltas so they line up with the layers
            # Look over all layers but the input layer, since it has no weights
            for l, layer in enumerate(self.layers[1:]):
                #  layer.values is the input to the layer (without activation function)
                weight_change = layer.values.T.dot(deltas[l]) * self.learning_rate
                layer.update_weights(weight_change, momentum)
        return error

    def update_learning_rate(self, epoch):
        p = 1e-3
        r = 10000
        self.learning_rate = p + (self.learning_rate - p) * (1 - (1 / (1 + np.exp(10 - (20 * epoch) / r))))

    def __str__(self):
        s = ""
        for l in self.layers:
            activation_function = ''
            if hasattr(l, 'activation_function'):
                if l.activation_function == Sigmoid:
                    activation_function = 'Sigmoid'
                elif l.activation_function == Relu:
                    activation_function = 'Relu'
                elif l.activation_function == TanH:
                    activation_function = 'TanH'
                elif l.activation_function == Linear:
                    activation_function = 'linear'

            s += str(l.outputs) + '-' + activation_function + '-'
        s += str(self.initial_learning_rate)
        return s


def split_data(sizes):
    if len(sizes) < 3:
        raise Exception('sizes must be of size 3')
    if np.sum(sizes) != 1:
        raise Exception('sizes must equal 1')
    data, mins_maxes = clean_and_read()
    np.random.shuffle(data)

    training_percentage = sizes[0]
    testing_percentage = sizes[1]
    # validataion_percentage = sizes[2]

    training_length = int(len(data) * training_percentage)
    testing_length = int(len(data) * testing_percentage)
    validation_length = len(data) - testing_length - training_length

    train = data[:training_length]
    test = data[training_length:training_length + testing_length]
    validation = data[training_length + testing_length:]

    return (train[:, :5], train[:, -1:]), (test[:, :5], test[:, -1:]), (validation[:, :5], validation[:, -1:]), mins_maxes


def test_validation(network, validation_data):
    errors = []
    for x, y in zip(validation_data[0], validation_data[1]):
        x = x.tolist()
        x += [1]
        prediction = network.run(np.array([x]))
        error = network.optimise(prediction, np.array([y]), momentum=False, update_weights=False)
        errors.append(error)
    return np.mean(np.abs(errors))


def train_training(network, training_data, epoch=0):
    errors = []
    for x, y in zip(training_data[0], training_data[1]):
        x = x.tolist()
        x += [1]
        prediction = network.run(np.array([x]))
        error = network.optimise(prediction, np.array([y]), momentum=False, update_weights=True)
        errors.append(error)
    network.update_learning_rate(epoch)
    return np.mean(np.abs(errors))


def generate_networks():
    networks = []
    learning_rates = [0.1, 0.01, 0.001]
    activation_functions = [Sigmoid, Relu, TanH, Linear]
    num_hidden = [4, 6, 8, 12, 16]

    for lr in learning_rates:
        for af1 in activation_functions:
            for af2 in activation_functions:
                for n in num_hidden:
                    i = InputLayer(6, 6)
                    h1 = HiddenLayer(6, n, af1)
                    o = OutputLayer(n, 1, af2)
                    network = Network(layers=[i, h1, o], learning_rate=lr)
                    networks.append(network)
    return networks


def run_network(n, train, test, validation, mins_maxes):
    training_last_mean_error = 10000
    validation_last_mean_error = 10000
    done = False
    all_training_errors = []
    all_validation_errors = []
    for e in range(1000000):
        error = train_training(n, train, epoch=e)
        if e % 100 == 0:
            training_mean_error = error
            # print("[{}] Error: {}".format(e, training_mean_error))
            validation_mean_error = test_validation(n, validation)
            all_training_errors.append(training_mean_error)
            all_validation_errors.append(validation_mean_error)
            if training_mean_error <= training_last_mean_error and validation_mean_error > validation_last_mean_error:
                print('stopping due to over fitting')
                done = True

            training_last_mean_error = training_mean_error
            validation_last_mean_error = validation_mean_error
        if done:
            break

    plt.plot(all_training_errors, label="Training error")
    plt.plot(all_validation_errors, label="Validation error")
    plt.ylabel('Absolute Mean Error')
    plt.xlabel('100\'s of Epochs')
    plt.title(str(n))
    plt.legend()
    plt.show()

    def convert_output(value):
        value = value[0][0]
        return np.interp(value, [0, 1], [mins_maxes[0]['PanE'], mins_maxes[1]['PanE']])

    test_x = test[0]
    test_y = test[1]
    errors = []
    for _x, _y in zip(test_x, test_y):
        _x = _x.tolist()
        _x += [1]
        prediction = n.run(np.array([_x]))

        # print('Prediction: {} | Actual: {}'.format(convert_output(prediction), convert_output([_y])))
        errors.append(n.optimise(prediction, np.array([_y]), momentum=False, update_weights=False))

    # print('Testing error: {}'.format(np.mean(np.abs(errors))))
    return np.mean(np.abs(errors))


pool = multiprocessing.Pool(16)


def multi_network_main():
    train, test, validation, mins_maxes = split_data((0.6, 0.2, 0.2))
    networks = generate_networks()
    jobs = []
    for n in networks:
        jobs.append(pool.apply_async(run_network, (n, train, test, validation, mins_maxes)))

    results = []
    for job, n in zip(jobs, networks):
        error = job.get(timeout=None)
        results.append((n, error))

    for n, error in results:
        print(n, error)

    results.sort(key=lambda x: x[1])

    best_result = results[0]

    print("Best network: {} | Best error: {}".format(best_result[1], best_result[0]))


def main():
    train, test, validation, mins_maxes = split_data((0.6, 0.2, 0.2))

    test_x = test[0]
    test_y = test[1]

    #  InputLayer takes the number of inputs and the number of outputs (always the same)
    l_i = InputLayer(6, 6)
    #  Hiddenlayer takes the number of inputs to the layer and the number of outputs
    l_h = HiddenLayer(6, 4, TanH)
    #  Output layer takes the number of inputs to the layer and the number of outputs
    l_o = OutputLayer(4, 1, Linear)
    n = Network(layers=[l_i, l_h, l_o], learning_rate=1e-1)

    training_last_mean_error = 0
    validation_last_mean_error = 0
    done = False
    all_training_errors = []
    all_validation_errors = []
    learning_rates = []
    for i in range(1000000):
        learning_rates.append(n.learning_rate)
        error = train_training(n, train, epoch=i)
        if i % 100 == 0:
            training_mean_error = error
            print("[{}] Error: {}".format(i, training_mean_error))
            validation_mean_error = test_validation(n, validation)
            all_training_errors.append(training_mean_error)
            all_validation_errors.append(validation_mean_error)
            if training_mean_error <= training_last_mean_error and validation_mean_error > validation_last_mean_error:
                print('stopping due to over fitting')
                done = True

            training_last_mean_error = training_mean_error
            validation_last_mean_error = validation_mean_error
        if done:
            break

    def convert_output(value):
        value = value[0][0]
        return np.interp(value, [0, 1], [mins_maxes[0]['PanE'], mins_maxes[1]['PanE']])

    predictions = []
    errors = []
    actual_errors = []
    mse_errors = []
    for _x, _y in zip(test_x, test_y):
        _x = _x.tolist()
        _x += [1]
        prediction = n.run(np.array([_x]))
        predictions.append([convert_output(prediction), convert_output([_y])])

        # print('Prediction: {} | Actual: {}'.format(convert_output(prediction), convert_output([_y])))
        actual_errors.append(convert_output([_y]) - convert_output(prediction))
        errors.append([_y] - prediction)
        mse_errors.append(n.optimise(prediction, np.array([_y]), momentum=False, update_weights=False))

    plt.plot(all_training_errors, label="Training error")
    plt.plot(all_validation_errors, label="Validation error")
    plt.ylabel('Absolute Mean Error')
    plt.xlabel('100\'s of Epochs')
    plt.title(str(n))
    plt.legend()
    plt.show()

    print('Testing error: {}'.format(np.mean(np.abs(errors))))
    print('Actual error: {}'.format(np.mean(np.abs(actual_errors))))
    print('MSE error: {}'.format(np.mean(np.abs(mse_errors))))


if __name__ == '__main__':
    # multi_network_main()
    main()

