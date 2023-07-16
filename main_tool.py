import numpy as np
import tensorflow as tf
from keras import layers, models
import os
import cv2
import pandas as pd

def create_convolutional_model():
    model = models.Sequential()
    model.add(layers.Conv2D(3, (3,3), padding="same", activation="relu", input_shape=(600, 600, 3)))
    model.add(layers.AveragePooling2D((10,10)))
    model.add(layers.Conv2D(3, (3,3), padding="same", activation="relu"))
    model.add(layers.AveragePooling2D((3,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(600, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model

def create_mlp_model():
    model = models.Sequential()

    model.add(layers.Dense(100, input_shape=(100,100,3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    
    return model


def load_data():
    folder1 = "data/resized"
    folder2 = "data/sepia"

    data = []
    for file in os.listdir(folder1):
        image = cv2.imread(folder1 + "/" + file)
        data.append([image, 0])

    for file in os.listdir(folder2):
        image = cv2.imread(folder2 + "/" + file)
        data.append([image,1])

    return data

def convolutional_model_run():
    data = load_data()
    model = create_convolutional_model()
    
    model.summary()


    indexes = [a for a in range(len(data))]
    trainPercentage = 0.75

    input_train = []
    output_train = []
    for i in indexes[:int(len(indexes) * trainPercentage)]:
        image = data[i][0]
        # resized_image = cv2.resize(image, (100,100))

        input_train.append(image)
        output_train.append(data[i][1])

    input_test = []
    output_test = []
    for i in indexes[int(len(indexes) * trainPercentage)+1:]:
        image = data[i][0]
        input_test.append(image)
        output_test.append(data[i][1])

    input_train = np.array(input_train)
    output_train = np.array(output_train)
    input_test = np.array(input_test)
    output_test = np.array(output_test)

    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    history = model.fit(input_train, output_train, epochs=10, validation_data=(input_test, output_test))

def simple_ann_model_run():
    class_names = ["normal", "sepia"]
    data = load_data()
    # model = create_convolutional_model()
    model = create_mlp_model()
    model.summary()


    indexes = [a for a in range(len(data))]
    trainPercentage = 0.75

    input_train = []
    output_train = []
    for i in indexes[:int(len(indexes) * trainPercentage)]:
        image = data[i][0]
        resized_image = cv2.resize(image, (100,100))

        input_train.append(np.asarray(resized_image))
        output_train.append(data[i][1])

    input_test = []
    output_test = []
    for i in indexes[int(len(indexes) * trainPercentage)+1:]:
        image = data[i][0]
        resized_image = cv2.resize(image, (100, 100))
        input_test.append(np.asarray(resized_image))
        output_test.append(data[i][1])

    input_train = np.array(input_train)
    output_train = np.array(output_train)
    input_test = np.array(input_test)
    output_test = np.array(output_test)

    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    history = model.fit(input_train, output_train, epochs=10, validation_data=(input_test, output_test))
    



if __name__ == "__main__":
   # convolutional_model_run()
   simple_ann_model_run()
