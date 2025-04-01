import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,Input,Flatten
import tensorflow as tf
import uuid
# define folder image
POS_PATH=os.path.join('data','position')
NEG_PATH=os.path.join('data','negative')
ANC_PATH=os.path.join('data','anchor')
# create folder
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PqATH)
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
    #flip frame 
    frame = cv2.flip(frame, 1)
    # show frame
    cv2.imshow('camshow',frame[130:130+250,180:180+250,:])
    # add image to positive folder 
    if cv2.waitKey(1) & 0XFF == ord('p'): 
       image=os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1()))
       cv2.imwrite(image,frame[130:130+250,180:180+250,:])
    # add image to anchor folder 
    if cv2.waitKey(1) & 0XFF == ord('a'): 
        image=os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(image,frame[130:130+250,180:180+250,:])
    # dress q to excape
    if cv2.waitKey(1) & 0XFF == ord('q'): 
        break
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()
# get 1000 image from folder
anchors= tf.data.Dataset.list_files(ANC_PATH+'\*jdg',1000)
positives= tf.data.Dataset.list_files(POS_PATH+'\*jdg',1000)
negatives= tf.data.Dataset.list_files(NEG_PATH+'\*jdg',1000)
# Preprocess resize image
def preProcessResize(path):
    #Convert jdg to bytecode
    imageByte=tf.io.read_file(path)
    #decode byte
    image=tf.io.decode_jpeg(imageByte)
    #resize 250x250 to 100x100
    reSizeImage=tf.image.resize(image,(100,100))
    #scale image to 0 and 1
    reSizeImage=reSizeImage/255.0
    #return
    return reSizeImage
# (anchor,positive) => 1, 1, 1, 1, 1
dataPositives=tf.data.Dataset(anchors,positives,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchors))))
# (anchor,negative) => 0, 0, 0, 0, 0
dataNegatives=tf.data.Dataset(anchors,negatives,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchors))))
# concatemate positives and negatives
data=dataPositives.concatenate(dataNegatives)
def preProcessRead(image_input,validation_img,label):
    return (preProcessResize(image_input),preProcessResize(validation_img),label)