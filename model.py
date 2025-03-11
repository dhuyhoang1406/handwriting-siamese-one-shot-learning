import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,Input,Flatten
import tensorflow as tf
# define folder image
POS_PATH=os.path.join('data','position')
NEG_PATH=os.path.join('data','negative')
ANC_PATH=os.path.join('data','anchor')
# create folder
    # os.makedirs(POS_PATH)
    # os.makedirs(NEG_PATH)
    # os.makedirs(ANC_PATH)
# moving data image to negative
    # for directory in os.listdir('lfw-deepfunneled'):
    #     dir_path = os.path.join('lfw-deepfunneled', directory)
    #     if os.path.isdir(dir_path):  
    #         for file in os.listdir(dir_path):
    #             EX_PATH = os.path.join(dir_path, file)
    #             NEW_PATH = os.path.join(NEG_PATH, file)
    #             os.replace(EX_PATH, NEW_PATH)
cap = cv2.VideoCapture(0)
#turn on webcam
while True:
    # read frame cam
    ret,frame = cap.read()
    # show frame
    cv2.imshow('camshow',frame[150:150+250,170:170+250,:])
    # dress q to excape
    if cv2.waitKey(1) & 0XFF == ord('q'): 
        break
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()




