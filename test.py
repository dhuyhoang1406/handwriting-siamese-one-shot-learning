import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

# Hàm custom để tính khoảng cách L1
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Load model từ file .keras hoặc .h5
siamese_model = tf.keras.models.load_model('siamese_model_final.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# siamese_model.summary()
# Hàm tiền xử lý ảnh
# def preprocess(file_path):
#     byte_img = tf.io.read_file(file_path)
#     img = tf.io.decode_jpeg(byte_img)
#     img = tf.image.resize(img, (100, 100))
#     img = img / 255.0
#     return img

# # Test: so sánh ảnh anchor và ảnh mới
# anchor_path = 'test/anchor.jpg'
# test_path = 'test/test1.jpg'

# anchor_img = preprocess(anchor_path)
# test_img = preprocess(test_path)

# # Dự đoán
# result = siamese_model.predict([tf.expand_dims(anchor_img, 0), tf.expand_dims(test_img, 0)])

# # Hiển thị kết quả
# print("Kết quả dự đoán:", result)
# if result > 0.5:
#     print("🟢 Khuôn mặt trùng khớp!")
# else:
#     print("🔴 Khuôn mặt KHÔNG trùng khớp!")
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
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

def preProcessRead(input_img, validation_img, label):
    input_img_resized = preProcessResize(input_img)
    validation_img_resized = preProcessResize(validation_img)
    
    return input_img_resized, validation_img_resized, label

data = data.map(preProcessRead)
data = data.cache()
data = data.shuffle(buffer_size=1024)

test_data = data.skip(int(len(data) * 0.7))
test_data = test_data.take(int(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
from tensorflow.keras.metrics import Precision, Recall

# Dữ liệu test đã chuẩn bị
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# Dự đoán
y_hat = siamese_model.predict([test_input, test_val])

# Xử lý lại kết quả dự đoán (Chuyển sang giá trị nhị phân)
y_hat_binary = [1 if prediction > 0.5 else 0 for prediction in y_hat]

# Tính các chỉ số
precision = precision_score(y_true, y_hat_binary)
recall = recall_score(y_true, y_hat_binary)
accuracy = accuracy_score(y_true, y_hat_binary)

# In kết quả
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')