import math

import numpy as np


class neural_network:

    def __init__(self, input_size: int, layers: [int], output_layer: int, activation, activDerivative):
        self.activation = activation
        self.activDerivative = activDerivative
        self.trainErr = 0
        self.correctGuesses = 0
        self.__no_layers = len(layers) + 1
        self.__out_size = output_layer

        if len(layers) > 0:
            self.__weights = [np.random.randn(input_size, layers[0])]
            self.__biases = [np.random.randn(layers[0])]
            for i in range(1, len(layers)):
                self.__weights.append(np.random.randn(layers[i - 1], layers[i]))
                self.__biases.append(np.random.randn(layers[i]))

            self.__weights.append(np.random.randn(layers[-1], output_layer))
            self.__biases.append(np.random.randn(output_layer))
        else:
            self.__weights = [np.random.randn(input_size, output_layer)]
            self.__biases = [np.random.randn(output_layer)]

    def softMax(self, input: np.array):
        expArr = np.exp(input)
        return expArr / sum(expArr)

    def train(self, input: np.array, output: np.array, learning_rate):
        h = [np.array(input)]
        a = [np.array(input)]

        for i in range(self.__no_layers - 1):
            h.append(np.dot(self.__weights[i].T, a[-1]) + self.__biases[i])
            a.append(self.activation(h[-1]))

        h.append(np.dot(self.__weights[-1].T, a[-1]) + self.__biases[-1])
        a.append(self.softMax(h[-1]))

        predicted_y = a[-1]
        expected_y = np.zeros(len(a[-1]))
        expected_y[output] = 1

        self.trainErr += self.loss(expected_y, predicted_y)
        self.correctGuesses += predicted_y.argmax() == output

        # backpropagation

        delta = predicted_y - expected_y
        delta = np.matrix(delta)
        dw = [np.matmul(delta.T, np.matrix(a[-2]))]
        db = [delta]

        for i in range(len(self.__weights) - 1, 0, -1):
            delta = np.dot(delta, self.__weights[i].T)
            delta = np.multiply(delta, self.activDerivative(h[i]))
            dw.insert(0, np.dot(delta.T, np.matrix(a[i - 1])))
            db.insert(0, delta)


        for i in range(self.__no_layers):
            self.__weights[i] -= learning_rate * dw[i].T
            self.__biases[i] -= np.array(learning_rate * np.array(db[i]))[0]

    def feedForward(self, inputs):
        h = [np.array(inputs)]
        a = [np.array(inputs)]

        self.trainErrr = 0

        for i in range(self.__no_layers - 1):
            h.append(np.dot(self.__weights[i].T, a[-1]) + self.__biases[i])
            a.append(self.activation(h[-1]))

        h.append(np.dot(self.__weights[-1].T, a[-1]) + self.__biases[-1])
        a.append(self.softMax(h[-1]))
        return a[-1]

    def loss(self, y, y_p):
        return -np.sum(y * np.log(y_p))
