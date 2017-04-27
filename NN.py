#!/usr/bin/python3
import numpy
import math
import csv


class NeuralNet:
    """"""

    def __init__(self, hidden_unit_count: int, momentum: float, learning_rate: float, filename: str,
                 training_inputs: numpy.array, training_targets: numpy.array,
                 testing_inputs: numpy.array, testing_targets: numpy.array,
                 uniform_weight=False):

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
        self.output_unit_count = 1
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

        self.hidden_units = [Node(self.number_of_inputs, uniform_weight) for x in range(self.hidden_unit_count)]
        # the + 1 is for the bias
        self.output_units = [Node(self.hidden_unit_count + 1, uniform_weight) for y in range(self.output_unit_count)]

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
        self.confusion_matrix = [[0 for x in range(self.output_unit_count)] for y in range(self.output_unit_count)]

    def train_epochs(self, number_of_epocs: int):
        """
        Trains the perceptrons a number_of_epocs times. Prints the accuracies at the end of each epoch
        :param number_of_epocs: The number of epocs to train over
        :return: None
        """
        # Print the initial accuracies
        training_accuracy, testing_accuracy = self.get_accuracy()
        self.file.write(str(training_accuracy) + ", " + str(testing_accuracy) + "\n")
        for i in range(number_of_epocs):
            self.train_epoch()
            training_accuracy, testing_accuracy = self.get_accuracy()
            self.file.write(str(training_accuracy) + ", " + str(testing_accuracy) + "\n")
        self.show_confusion()

    def get_accuracy(self):
        """
        Calculates accuracy of training data and testing data
        Updates confusion matrix
        :return: accuracy of training data, accuracy of testing data
        """
        correct_training_count = 0
        correct_testing_count = 0
        self.reset_confusion()

        for i in range(self.training_count):
            hidden_activations, output_value, output = self.propagate(self.training_inputs[i])
            if output_value == self.training_targets[i]:
                correct_training_count += 1

        for i in range(self.testing_count):
            hidden_activations, output_value, output = self.propagate(self.testing_inputs[i])
            if output_value == self.testing_targets[i]:
                correct_testing_count += 1
            # TODO: this is commented out for testing self.confusion_matrix[self.testing_targets[i]][output_value] += 1

        return correct_training_count/self.training_count, correct_testing_count/self.testing_count



    def show_confusion(self):
        """shows the confusion matrix"""
        for r in self.confusion_matrix:
            for c in r:
                self.file.write(str(c) + ", ")
            self.file.write("\n")

    def train_epoch(self):
        for i in range(self.training_count):
            hidden_activations, output_value, output = self.propagate(self.training_inputs[i])

            # calculate error values
            # TODO: this is for after testing target = self.value_to_list[self.training_targets[i]]
            target = self.training_targets[i]
            output_error = self.get_output_error(output, target)
            hidden_error = self.get_hidden_error(hidden_activations, output_error)

            # update weights
            self.update_hidden_to_output_weights(hidden_activations, output_error)
            self.update_input_to_hidden_weights(self.training_inputs[i], hidden_error)

    def propagate(self, inputs):
        hidden_activations = self.propagate_input_to_hidden(inputs)
        output_value, output = self.propagate_hidden_to_output(hidden_activations)
        return hidden_activations, output_value, output


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
        got_error = False
        if target == 0:
            target = .1
            if output < .1:
                output_error.append(0)
                got_error = True
        elif target == 1:
            target = .9
            if output > .9:
                output_error.append(0)
                got_error = True
        if not got_error:
            output_error.append(output * (1 - output) * (target - output))
        return numpy.array(output_error)
        #TODO: uncomment after testing for k in range(self.output_unit_count):
            #if target[k] == .1 and output[k] < .1:
                #output_error.append(0)
            #elif target[k] == .9 and output[k] > .9:
                #output_error.append(0)
            #else:
                #output_error.append(output[k] * (1 - output[k]) * (target[k] - output[k]))
        #return numpy.array(output_error)

    def get_hidden_error(self, hidden_activations, output_errors):
        hidden_errors = []
        for j in range(1, self.hidden_unit_count + 1):
            hidden_errors.append(hidden_activations[j] * (1 - hidden_activations[j])
                                 * self.sum_error_effect(j, output_errors))
        return numpy.array(hidden_errors)

    def sum_error_effect(self, hidden_unit_index, output_errors):
        error_effect = 0
        #for k in range(self.output_unit_count):
            # k + 1 skips the weight of the bias
            # TODO: uncomment after testing error_effect += output_unit.weights[k] * output_errors[k]
        for k in range(self.output_unit_count):
            error_effect += self.output_units[k].weights[hidden_unit_index] * output_errors[k]

        return error_effect

    def update_hidden_to_output_weights(self, hidden_activations, output_error):
            for k in range(self.output_unit_count):
                self.output_units[k].update_weights(self.learning_rate, self.momentum, hidden_activations, output_error[k])


    def update_input_to_hidden_weights(self, inputs, hidden_error):
        for k in range(self.hidden_unit_count):
            self.hidden_units[k].update_weights(self.learning_rate, self.momentum, inputs, hidden_error[k])




class Node:
    """Is for holding the state of weights, providing output when given input, """

    def __init__(self, size, weight):
        self.size = size
        if not weight:
            self.weights = self.starting_weights()
        else:
            self.weights = numpy.array([weight for x in range(self.size)])
        self.weight_change = numpy.zeros(size)

    def starting_weights(self):
        weights = numpy.random.rand(1, self.size) - 0.5
        return weights

    def sigmoid_activation(self, inputs):
        dot_product = numpy.dot(self.weights, inputs)
        sigmoid_activation_value = 1 / (1 + math.exp(-dot_product))
        return sigmoid_activation_value

    def update_weights(self, learning_rate: float, momentum: float, unit_outputs, error):
        self.weight_change = learning_rate * error * unit_outputs + momentum * self.weight_change
        self.weights = self.weights + self.weight_change


def read_data(filename: str):
    """Massages the data from csv into numpy arrays"""
    print("Loading " + filename)
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

    #TODO uncomment this after testing data = data / 255
    data = numpy.c_[numpy.ones((len(data), 1)), data]
    print("Done")

    return targets, data

if __name__ == "__main__":
    training_targets_param, training_inputs_param = read_data("./test.csv")
    testing_targets_param, testing_inputs_param = read_data("./test.csv")
    learning_rate_param = 0.2
    hidden_unit_count_param = 2
    momentum_param = 0.9
    filename_param = "./" + str(learning_rate_param) + ".txt"
    neural_net = NeuralNet(hidden_unit_count_param, momentum_param, learning_rate_param, filename_param,
                           training_inputs_param, training_targets_param, testing_inputs_param, testing_targets_param,
                           0.1)
    neural_net.train_epochs(1)
    neural_net.file.close()
