import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps
import os
import ssl
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as ts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
from PIL import Image
from sqlalchemy import true

#TRAIN SESSION CODE
#X = the image of the number, y = the number

x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nClasses = len(classes)

#SPLITING
trainX, testX, trainY, testY = ts(x, y, train_size = 3500, test_size = 500)

trainX = trainX/255.0
testX = testX/255.0

clf = lr(solver = "saga", multi_class = "multinomial")
clf.fit(trainX, trainY)

predict = clf.predict(testX)
print(accuracy_score(testY, predict), "%")


#CAMERA CAPTURE
cap = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        height, width = gray.shape
        upperLeft = (int(width/2 - 36), int(height/2 - 36))
        bottomRight = (int(width/2 + 36), int(height/2 + 36))

        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 7)
        roi = gray[upperLeft[1]: bottomRight[1], upperLeft[0]: bottomRight[0]]

        
        img_PIL = Image.fromarray(roi)
        img_bw = img_PIL.convert("L")
        img_resize = img_bw.resize((28, 28))
        final_img = PIL.ImageOps.invert(img_resize)
        
        
        pixelFilter = 20
        
        min_pixel = np.percentile(final_img, pixelFilter)
        scaledImg = np.clip(final_img - min_pixel, 0 , 255)

        max_pixel = np.max(final_img)
        scaledImg1 = np.asarray(scaledImg)/max_pixel

        test_Sample = np.array(scaledImg1).reshape(1,784)
        testPred = clf.predict(test_Sample)
        print("Predicted value: ", testPred)

        cv2.imshow("FRAME", gray)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        pass


