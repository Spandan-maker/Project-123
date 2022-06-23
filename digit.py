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

x,y = fetch_openml("mnist_784", version = 1, return_X_y = True)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nClasses = len(classes)

#SPLITING
trainX, testX, trainY, testY = ts(x, y, train_size = 7500, test_size = 2500)

trainX = trainX/255.0
testX = testX/255.0

clf = lr(solver = "saga", multi_class = "multinomial").fit(trainX, trainY)
predict = clf.predict(testX)
#print(accuracy_score(testY, predict), "%")


#CAMERA CAPTURE
cap = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        height, width = gray.shape
        upperLeft = (int(width/2 - 70), int(height/2 - 70))
        bottomRight = (int(width/2 + 70), int(height/2 + 70))

        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        roi = gray[upperLeft[1]: bottomRight[1], upperLeft[0]: bottomRight[0]]
        
        img_PIL = Image.fromarray(roi)
        img_bw = img_PIL.convert("L")
        img_resize = img_bw.resize((28, 28), Image.ANTIALIAS)
        final_img = PIL.ImageOps.invert(img_resize)
        
        
        pixelFilter = 20
        min_pixel = np.percentile(final_img, pixelFilter)
        max_pixel = np.max(final_img)

        scaledImg1 = np.clip(final_img - min_pixel, 0 , 255)
        scaledImg2 = np.asarray(scaledImg1)/max_pixel

        #testSample = np.array(scaledImg2).reshape(1, 784)
        #testPredict = clf.predict(testSample)
        #print("This is the test predict value: ", testPredict)

        scaledImg = cv2.resize(scaledImg2, (960, 540))
        cv2.imshow("Digit Recognition", scaledImg)


        if cv2.waitKey(1) & 0xFF == ord("Q"):
            break

    except Exception as e:
        pass


