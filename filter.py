import os

from PIL import Image
from os import listdir
from alive_progress import alive_bar
import cv2 as cv2
import numpy as np

with alive_bar(len(listdir("data/resized"))) as bar:
    for file in listdir("data/resized"):
        if file.count(".jpg") > 0 or file.count(".jpeg") > 0:
            image = cv2.imread("data/resized/" + file)

            sepiaImg = np.array(image, dtype=np.float64)
            sepiaImg = cv2.transform(sepiaImg, np.matrix([[0.272, 0.543, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
            sepiaImg[np.where(sepiaImg > 255)] = 255
            sepiaImg = np.array(sepiaImg, dtype=np.uint8)

            cv2.imwrite("data/sepia/"+file, sepiaImg)
        bar()
