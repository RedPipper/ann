import math
import random

import sklearn.metrics
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from alive_progress import alive_bar

from neural_network import neural_network
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


def sigmoid(x):
    return math.exp(x) / (1 + math.exp(x))


def scaleMinMax(data):
    minim = min(data)
    maxim = max(data)
    return [(dp - minim) / (maxim - minim) for dp in data]


def normaliseFeature01(feature):
    minim = min(feature)
    maxim = max(feature)

    return scaleMinMax(feature)


def ReLU(H):
    return np.array([max(0.0, a) for a in H])


def ReLU_deriv(H):
    return np.array([1 if a > 0 else 0 for a in H])


def SIGMOID(H):
    return np.array([1 / (1 + math.exp(-a)) for a in H])


def SIGMOID_deriv(H):
    return np.array([(1 / (1 + math.exp(-a)) ** 2) * (math.exp(-a)) for a in H])


def train_ANN(net: neural_network, input_sample, output_sample, iterations, epochs, learningRate):
    for e in range(epochs):

        with alive_bar(iterations) as bar:
            for k in range(iterations):
                for i in range(len(input_sample)):
                    net.train(input_sample[i], output_sample[i], learningRate)
                bar.text(f'# Epoch {e + 1}')
                bar()

        print(
            f"Average of correct guesses over training was {net.correctGuesses / (iterations * (e + 1))}/{len(input_sample)}")

    return int((net.correctGuesses / (iterations * epochs)) / len(input_sample) * 100)


def train_test_digits():
    (data, target) = load_digits(return_X_y=True)

    data = [normaliseFeature01(a) for a in data]
    # print(data)
    network = neural_network(len(data[0]), [128], 10, ReLU, ReLU_deriv)
    train_test_percentage = 0.75

    # data separation
    input_train = data[:int(len(data) * train_test_percentage)]
    output_train = target[:int(len(data) * train_test_percentage)]
    input_test = data[int(len(data) * train_test_percentage) + 1:]
    output_test = target[int(len(data) * train_test_percentage) + 1:]

    train_err = train_ANN(network, input_train, output_train, 100, 2, 0.1)

    correctGuesses = 0
    all_outs = []
    for i in range(len(output_test)):
        outputs = network.feedForward(input_test[i])
        correctGuesses += outputs.argmax() == output_test[i]
        all_outs.append(outputs.argmax())

    print(f'Correct guesses of tests: {correctGuesses}/{len(input_test)}')
    print(f'Error rate over training: {train_err}%')
    print(f'Error rate over tests: {int(correctGuesses / len(input_test) * 100)}%')
    print(f'Fitting parameter (smaller the better): {abs(train_err - int(correctGuesses / len(input_test) * 100))}')
    print("\n\n")
    ConfusionMatrixDisplay.from_predictions(output_test, all_outs, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def train_test_irisi():
    dataFrame = pd.read_csv("data/iris.data")
    labels = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    dataFrame.columns = labels
    dataFrame["class"] = [0 if a == "Iris-setosa" else 1 if a == "Iris-versicolor" else 2 for a in dataFrame["class"]]

    data = []
    target = []
    for i in range(len(dataFrame)):
        data.append([dataFrame["sepal length"][i], dataFrame["sepal width"][i], dataFrame["petal length"][i],
                     dataFrame["petal width"][i]])
        target.append(dataFrame["class"][i])

    indexes = [a for a in range(len(data))]

    train_test_percentage = 0.8
    classes = [[], [], []]

    for i in range(len(data)):
        classes[target[i]].append(data[i])

    indexes = [a for a in range(len(classes[0]))]
    random.shuffle(indexes)

    input_train = []
    output_train = []
    for i in indexes[:int(len(indexes) * train_test_percentage)]:
        input_train.append(classes[0][i])
        input_train.append(classes[1][i])
        input_train.append(classes[2][i])

        output_train.append(0)
        output_train.append(1)
        output_train.append(2)

    input_test = []
    output_test = []
    for i in indexes[int(len(indexes) * train_test_percentage) + 1:]:
        input_test.append(classes[0][i])
        input_test.append(classes[1][i])
        input_test.append(classes[2][i])

        output_test.append(0)
        output_test.append(1)
        output_test.append(2)

    network = neural_network(len(data[0]), [], 3, SIGMOID, SIGMOID_deriv)
    train_err = train_ANN(network, input_train, output_train, 100, 5, 0.1)

    correctGuesses = 0
    all_outs = []
    for i in range(len(output_test)):
        outputs = network.feedForward(input_test[i])
        correctGuesses += outputs.argmax() == output_test[i]
        all_outs.append(outputs)
    print(f'Correct guesses of tests: {correctGuesses}/{len(input_test)}')
    print(f'Success rate over training: {train_err}%')
    print(f'Success rate over tests: {int(correctGuesses / len(input_test) * 100)}%')
    print(f'Fitness parameter:{abs((100 - train_err) - (100 - int(correctGuesses / len(input_test) * 100)))}')
    print("\n\n")

    ConfusionMatrixDisplay.from_predictions(output_test, [a.argmax() for a in all_outs],
                                            display_labels=["Iris Setosa", "Iris Versicolour", "Irist Virginica"])


if __name__ == "__main__":
    train_test_digits()
    # train_test_irisi()
