#!/usr/bin/python3
import numpy
import math
import csv


class NeuralNet:
    """"""

    def __init__(self, hidden_unit_count: int, momentum: float, learning_rate: float, filename: str,
                 training_inputs: numpy.array, training_targets: numpy.array,
                 testing_inputs: numpy.array, testing_targets: numpy.array,):

        """
        :param training_inputs: a two dimensional numpy.array of training input vaules
        normalized to be between 0 and 1 for training
        :param training_targets: a one dimensional numpy.array of target values between 0 and 9 for training
        :param testing_inputs: a two dimensional numpy.array of training input vaules 
        normalized to be between 0 and 1 for testing
        :param testing_targets: a one dimensiona lnumpy.array of target values between 0 and 9 for training
        :param learning_rate: a float that governs how fast the weights are changed
        """
        self.hidden_unit_count = hidden_unit_count
        self.output_unit_count = 10
        self.momentum = momentum
        self.training_count = len(training_inputs)
        self.training_inputs = training_inputs
        self.training_targets = training_targets

        self.testing_count = len(testing_inputs)
        self.testing_inputs = testing_inputs
        self.testing_targets = testing_targets
        self.learning_rate = learning_rate
        self.filename = filename
        self.file = open(filename, 'w')

        self.number_of_inputs = training_inputs.shape[1]
        # a 10x10 matrix to keep track of confused digits
        # self.confusion_matrix = numpy.zeros((10, 10))
        self.confusion_matrix = []
        self.reset_confusion()

        self.hidden_units = [Node(self.number_of_inputs) for x in range(self.hidden_unit_count)]
        # the + 1 is for the bias
        self.output_units = [Node(self.hidden_unit_count + 1) for y in range(self.output_unit_count)]

        self.value_to_list = {0: [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                              1: [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                              2: [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                              3: [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                              4: [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
                              5: [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
                              6: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
                              7: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1],
                              8: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
                              9: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
                              }

    def reset_confusion(self):
        self.confusion_matrix = [[0 in range(self.output_unit_count)] in range(self.output_unit_count)]

    def train_epoch(self):
        for i in range(self.training_count):
            # get output from hidden layer
            hidden_activations = self.propagate_input_to_hidden(self.training_inputs[i])
            # get output from output layer
            output_value, output = self.propagate_hidden_to_output(hidden_activations)

            # calculate error values
            target = self.value_to_list[self.training_targets[i]]
            output_error = self.get_output_error(output, target)
            hidden_error = self.get_hidden_error(hidden_activations, output_error)

            # update weights
            self.update_hidden_to_output_weights(hidden_activations, output_error)
            self.update_input_to_hidden_weights(hidden_error)

    def propagate_input_to_hidden(self, inputs):
        hidden_outputs = []
        for unit in self.hidden_units:
            hidden_outputs.append(unit.sigmoid_activation(inputs))
        # add the bias for the output layer
        hidden_outputs.insert(0, 1)
        return numpy.array(hidden_outputs)

    def propagate_hidden_to_output(self, hidden_activations):
        output = []
        for unit in self.output_units:
            output.append(unit.sigmoid_activation(hidden_activations))
        output = numpy.array(output)
        return output.argmax(), output

    def get_output_error(self, output, target):
        output_error = []
        for k in range(self.output_unit_count):
            if target == .1 and output[k] < .1:
                output_error.append(0)
            elif target == .9 and output[k] > .9:
                output_error.append(0)
            else:
                output_error.append(output[k] * (1 - output[k]) * (target[k] - output[k]))
        return numpy.array(output_error)

    def get_hidden_error(self, hidden_activations, output_errors):
        hidden_errors = []
        for j in range(self.hidden_unit_count):
            hidden_errors.append(hidden_activations[j] * (1 - hidden_activations[j])
                                 * self.sum_error_effect(self.hidden_units, output_errors))
        return numpy.array(hidden_errors)

    def sum_error_effect(self, output_unit, output_errors):
        error_effect = 0
        for k in range(self.output_unit_count):
            # k + 1 skips the weight of the bias
            error_effect += output_unit.weights[k] * output_errors[k]
        return error_effect

    def update_hidden_to_output_weights(self, hidden_activations, output_error):
        for j in range(self.hidden_unit_count):
            for k in range(self.output_unit_count):
                self.hidden_units[j].weights[k] = self.learning_rate * output_error[k] * hidden_activations[j]


    def update_input_to_hidden_weights(self, hidden_error):
        pass


class Node:
    """Is for holding the state of weights, providing output when given input, """

    def __init__(self, size):
        self.size = size
        self.weights = self.starting_weights()

    def starting_weights(self):
        weights = numpy.random.rand(1, self.size) - 0.5
        return weights

    def sigmoid_activation(self, inputs):
        return 1 / (1 + math.exp(numpy.dot(self.weights, inputs)))

    def update_weights(self, output, target, inputs, learning_rate):
        weight_change = inputs * learning_rate * (target - output)
        self.weights = self.weights + weight_change


def read_data(filename: str):
    """Massages the data from csv into numpy arrays"""
    print("Loading " + filename)
    # inputs = pandas.read_csv(filename, header=None, index_col=0) / 255
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    temp = data
    data = []
    for row in temp:
        new_row = []
        for char in row:
            new_row.append(int(char))
        data.append(new_row)
    data = numpy.array(data)
    targets = data[:, 1]
    data = data[:, 1:]
    data = data / 255
    data = numpy.c_[numpy.ones((len(data), 1)), data]
    print("Done")

    return targets, data

if __name__ == "__main__":
    training_targets_param, training_inputs_param = read_data("./micro.csv")
    testing_targets_param, testing_inputs_param = read_data("./micro.csv")
    learning_rate_param = 0.1
    hidden_unit_count_param = 2
    momentum_param = 0.9
    filename_param = "./" + str(learning_rate_param) + ".txt"
    neural_net = NeuralNet(hidden_unit_count_param, momentum_param, learning_rate_param, filename_param,
                           training_inputs_param, training_targets_param, testing_inputs_param, testing_targets_param, )
    neural_net.train_epoch()
    neural_net.file.close()
