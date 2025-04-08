import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,Input,Flatten
import tensorflow as tf
import uuid
import time
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
count = 0
#turn on webcam
while True:
    # read frame cam
    ret,frame = cap.read()
    #flip frame 
    frame = cv2.flip(frame, 1)
    # show frame
    roi = frame[130:130+250, 180:180+250, :]
    cv2.imshow('camshow', roi)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p') or key == ord('a'):
        folder_path = POS_PATH if key == ord('p') else ANC_PATH
        print(f"Starting capture to {'POS_PATH' if key == ord('p') else 'ANC_PATH'}...")

        for i in range(500):
            ret, frame = cap.read()  # Cập nhật frame mới
            frame = cv2.flip(frame, 1)  # Đảo ngược khung hình
            roi = frame[130:130+250, 180:180+250, :]
            
            image_path = os.path.join(folder_path, f"{uuid.uuid1()}.jpg")
            cv2.imwrite(image_path, roi)
            print(f"Saved {i+1}/500 to {folder_path}")

            time.sleep(0.1)  # Chờ 3 giây trước khi chụp tiếp

    if key == ord('q'):  # Thoát chương trình
        break
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()
# get 1000 image from folder
anchors= tf.data.Dataset.list_files(ANC_PATH+'\*jdg',500)
positives= tf.data.Dataset.list_files(POS_PATH+'\*jdg',500)
negatives= tf.data.Dataset.list_files(NEG_PATH+'\*jdg',500)
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