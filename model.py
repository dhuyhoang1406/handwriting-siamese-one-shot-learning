import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import uuid
import time
from tensorflow.keras.metrics import Precision, Recall

# define folder image
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

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

# cap = cv2.VideoCapture(0)
# count = 0
# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     roi = frame[130:130+250, 180:180+250, :]
#     cv2.imshow('camshow', roi)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('p') or key == ord('a'):
#         folder_path = POS_PATH if key == ord('p') else ANC_PATH
#         print(f"Starting capture to {'POS_PATH' if key == ord('p') else 'ANC_PATH'}...")

#         for i in range(500):
#             ret, frame = cap.read()
#             frame = cv2.flip(frame, 1)
#             roi = frame[130:130+250, 180:180+250, :]
#             image_path = os.path.join(folder_path, f"{uuid.uuid1()}.jpg")
#             cv2.imwrite(image_path, roi)
#             print(f"Saved {i+1}/500 to {folder_path}")
#             time.sleep(0.1)

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# get 1000 image from folder
anchors = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg', shuffle=False)
positives = tf.data.Dataset.list_files(POS_PATH + '/*.jpg', shuffle=False)
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(500)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(500)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(500)

# Preprocess resize image
def preProcessResize(path):
    imageByte = tf.io.read_file(path)
    image = tf.io.decode_jpeg(imageByte)
    reSizeImage = tf.image.resize(image, (100, 100))
    reSizeImage = reSizeImage / 255.0
    return reSizeImage

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# samples = data.as_numpy_iterator()
# exampple = samples.next()
# exampple

# Preprocessing function
def preProcessRead(input_img, validation_img, label):
    input_img_resized = preProcessResize(input_img)
    validation_img_resized = preProcessResize(validation_img)
    
    # Print the shape and some values to debug
    # print(f"Input Image Shape: {input_img_resized.shape}")
    # print(f"Validation Image Shape: {validation_img_resized.shape}")
    # print(f"Validation Image Sample Values (Min, Max): {np.min(validation_img_resized), np.max(validation_img_resized)}")
    
    return input_img_resized, validation_img_resized, label

# Now, plot the image (checking after preprocessing)
# res = preProcessRead(*exampple)
# plt.imshow(res[1])  # Display the validation image
# plt.show()

# res[2]
# Build dataloader 
data = data.map(preProcessRead)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training partition
train_data = data.take(int(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(int(len(data) * 0.7))
test_data = test_data.take(int(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Make embedding model
def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=inp, outputs=d1, name='embedding')


embedding = make_embedding()
# embedding.summary()

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Build Siamese Model
def make_siamese_model(): 
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
# siamese_model.summary()
# Set up loss and optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
siamese_model = make_siamese_model()
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Build train step function
@tf.function
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss


# Training loop
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)

            
EPOCHS = 5
train(train_data, EPOCHS)