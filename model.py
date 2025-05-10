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

def load_image_paths_and_labels(dataset_path):
    image_paths = []
    labels = []
    label = 0

    if not os.path.exists(dataset_path):
        print(f"Thư mục {dataset_path} không tồn tại.")
        return [], []

    for character_folder in os.listdir(dataset_path):
        character_folder_path = os.path.join(dataset_path, character_folder)
        if os.path.isdir(character_folder_path):
            for character_subfolder in range(1, 21):
                character_subfolder_name = f"character{character_subfolder:02d}"
                character_subfolder_path = os.path.join(character_folder_path, character_subfolder_name)

                if os.path.isdir(character_subfolder_path):
                    for image_file in os.listdir(character_subfolder_path):
                        if image_file.endswith('.png'):
                            image_path = os.path.join(character_subfolder_path, image_file)
                            image_paths.append(image_path.encode())  # lưu dưới dạng bytes như yêu cầu
                            labels.append(label)
            label += 1
    return image_paths, labels

def create_pairs_from_paths(image_paths, labels):
    label_to_paths = {}
    for path, label in zip(image_paths, labels):
        label = int(label)
        label_to_paths.setdefault(label, []).append(path)

    anchors, positives, negatives = [], [], []

    label_list = list(label_to_paths.keys())

    for label in label_list:
        same_class_paths = label_to_paths[label]
        if len(same_class_paths) < 2:
            continue

        for i in range(len(same_class_paths) - 1):
            anchor = same_class_paths[i]
            positive = same_class_paths[i + 1]
            anchors.append(anchor)
            positives.append(positive)

            # Chọn lớp khác
            neg_label = random.choice([l for l in label_list if l != label])
            negative = random.choice(label_to_paths[neg_label])
            negatives.append(negative)

    return anchors, positives, negatives

def create_tf_dataset(anchors, positives, negatives):
    anchor_ds = tf.data.Dataset.from_tensor_slices(anchors)
    positive_ds = tf.data.Dataset.from_tensor_slices(positives)
    negative_ds = tf.data.Dataset.from_tensor_slices(negatives)

    # Positive pairs (label 1)
    positive_pairs = tf.data.Dataset.zip((
        anchor_ds,
        positive_ds,
        tf.data.Dataset.from_tensor_slices(tf.ones(len(anchors)))
    ))

    # Negative pairs (label 0)
    negative_pairs = tf.data.Dataset.zip((
        anchor_ds,
        negative_ds,
        tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchors)))
    ))

    # Gộp lại và shuffle ngay
    data = positive_pairs.concatenate(negative_pairs)
    data = data.shuffle(buffer_size=2 * len(anchors))  # shuffle toàn bộ
    return data

# Đường dẫn cho train và test dataset
train_dataset_path = "\handwriting-siamese-one-shot-learning\Omniglot Dataset\images_background"
test_dataset_path = "\handwriting-siamese-one-shot-learning\Omniglot Dataset\images_evaluation"

# B1: Load train data từ images_background
train_image_paths, train_labels = load_image_paths_and_labels(train_dataset_path)
train_anchors, train_positives, train_negatives = create_pairs_from_paths(train_image_paths, train_labels)
train_data = create_tf_dataset(train_anchors, train_positives, train_negatives)

# B2: Load test data từ images_evaluation
test_image_paths, test_labels = load_image_paths_and_labels(test_dataset_path)
test_anchors, test_positives, test_negatives = create_pairs_from_paths(test_image_paths, test_labels)
test_data = create_tf_dataset(test_anchors, test_positives, test_negatives)

# B3: Lấy một sample từ train_data và test_data để kiểm tra
train_samples = train_data.as_numpy_iterator()
test_samples = test_data.as_numpy_iterator()

# Lấy 1 ví dụ từ train_data
train_example = next(train_samples)
print("Train Example:", train_example)

# Lấy 1 ví dụ từ test_data
test_example = next(test_samples)
print("Test Example:", test_example)

class DatasetInfoPrinter:
    def __init__(self, dataset_name, image_paths, labels, anchors, positives, negatives):
        self.dataset_name = dataset_name
        self.total_images = len(image_paths)
        self.total_classes = len(set(labels))
        self.total_positive_pairs = len(anchors)
        self.total_negative_pairs = len(negatives)
        self.total_pairs = self.total_positive_pairs + self.total_negative_pairs

    def print_info(self):
        print(f"--- {self.dataset_name.upper()} ---")
        print(f"• Số lượng ảnh: {self.total_images:,}")
        print(f"• Số lượng lớp/ký tự: {self.total_classes}")
        print(f"• Số cặp dương: {self.total_positive_pairs:,}")
        print(f"• Số cặp âm: {self.total_negative_pairs:,}")
        print(f"• Tổng số cặp huấn luyện: {self.total_pairs:,}")
        print()

# In thông tin tập train
train_info = DatasetInfoPrinter("Tập huấn luyện", train_image_paths, train_labels, train_anchors, train_positives, train_negatives)
train_info.print_info()

# In thông tin tập test
test_info = DatasetInfoPrinter("Tập kiểm tra", test_image_paths, test_labels, test_anchors, test_positives, test_negatives)
test_info.print_info()

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_png(byte_img, channels=1)  # Đọc ảnh dưới dạng grayscale
    img = tf.image.grayscale_to_rgb(img)  # Chuyển từ grayscale sang RGB
    img = tf.image.resize(img, (100, 100))  # Thay đổi kích thước ảnh
    img = img / 255.0  # Chuẩn hóa ảnh

    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# Áp dụng lại pipeline
train_data = train_data.map(preprocess_twin)
train_data = train_data.cache()
train_data = train_data.batch(16)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

test_data = test_data.map(preprocess_twin)
test_data = test_data.cache()
test_data = test_data.batch(16)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

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


    return Model(inputs=[inp], outputs=d1, name='embedding')

embedding = make_embedding()

# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
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

# Optimizer và loss
opt = tf.keras.optimizers.Adam(1e-5)
binary_cross_loss = tf.losses.BinaryCrossentropy()

# Checkpoint
checkpoint_dir = '/content/drive/MyDrive/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Tự động load checkpoint nếu có
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
start_epoch = 1
if latest_checkpoint:
    print(f"🔁 Loading weights from: {latest_checkpoint}")
    checkpoint.restore(latest_checkpoint)
    # Parse số epoch từ tên checkpoint (nếu đặt đúng định dạng)
    ckpt_name = os.path.basename(latest_checkpoint)
    if ckpt_name.startswith('ckpt-'):
        start_epoch = int(ckpt_name.split('-')[-1]) + 1  # tiếp tục từ epoch kế tiếp

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss, yhat

# Theo dõi lịch sử
history = {
    "loss": [],
    "recall": [],
    "precision": []
}

def train(data, EPOCHS, start_epoch=start_epoch):
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        start_time = time.time()

        r = Recall()
        p = Precision()
        total_loss = 0.0
        batches = 0
        total_steps = len(data)

        for idx, batch in enumerate(data):
            loss, yhat = train_step(batch)
            total_loss += loss.numpy()
            batches += 1

            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)

            progress = int((idx + 1) / total_steps * 30)
            bar = '━' * progress + '-' * (30 - progress)

            avg_loss = total_loss / batches
            recall_val = r.result().numpy()
            precision_val = p.result().numpy()

            elapsed = time.time() - start_time
            time_per_step = elapsed / (idx + 1)

            print(f"\r{idx+1}/{total_steps} ━{bar}━ "
                  f"{int(elapsed)}s {int(time_per_step * 1000)}ms/step "
                  f"- loss: {avg_loss:.4f} - recall: {recall_val:.4f} - precision: {precision_val:.4f}", end='')

        # Lưu checkpoint mỗi 2 epoch
        if epoch % 2 == 0:
            path = checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"\n💾 Saved checkpoint to: {path}")

        history["loss"].append(total_loss / batches)
        history["recall"].append(r.result().numpy())
        history["precision"].append(p.result().numpy())

    return history

history = train(train_data, EPOCHS=20)

epochs_range = range(1, len(history["loss"]) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, history["loss"], label='Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs_range, history["recall"], label='Recall', color='orange')
plt.title("Recall")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs_range, history["precision"], label='Precision', color='green')
plt.title("Precision")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)

plt.tight_layout()
plt.show()