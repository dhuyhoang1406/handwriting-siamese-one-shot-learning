import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer

# Hàm load dữ liệu ảnh từ thư mục
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

# Hàm tạo cặp ảnh
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

# Tạo dataset từ các cặp ảnh
def create_tf_dataset(anchors, positives, negatives):
    anchor_ds = tf.data.Dataset.from_tensor_slices(anchors).map(lambda x: preprocess(x))
    positive_ds = tf.data.Dataset.from_tensor_slices(positives).map(lambda x: preprocess(x))
    negative_ds = tf.data.Dataset.from_tensor_slices(negatives).map(lambda x: preprocess(x))

    # Positive pairs (label 1)
    positive_pairs = tf.data.Dataset.zip((anchor_ds, positive_ds, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchors)))))
    
    # Negative pairs (label 0)
    negative_pairs = tf.data.Dataset.zip((anchor_ds, negative_ds, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchors)))))

    # Gộp lại và shuffle ngay
    data = positive_pairs.concatenate(negative_pairs)
    data = data.shuffle(buffer_size=2 * len(anchors))  # shuffle toàn bộ
    return data

# Preprocess ảnh
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_png(byte_img, channels=1)  # Đọc ảnh dưới dạng grayscale
    img = tf.image.grayscale_to_rgb(img)  # Chuyển từ grayscale sang RGB
    img = tf.image.resize(img, (100, 100))  # Thay đổi kích thước ảnh
    img = img / 255.0  # Chuẩn hóa ảnh
    return img

# Lớp tính khoảng cách L1 trong mô hình Siamese
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Đường dẫn dữ liệu test
test_dataset_path = "D:/Du_AN/NCKH/handwriting-siamese-one-shot-learning/Omniglot Dataset/images_evaluation"

# B2: Load test data từ images_evaluation
test_image_paths, test_labels = load_image_paths_and_labels(test_dataset_path)
test_anchors, test_positives, test_negatives = create_pairs_from_paths(test_image_paths, test_labels)
test_data = create_tf_dataset(test_anchors, test_positives, test_negatives)

# Load mô hình Siamese đã huấn luyện
siamese_model = tf.keras.models.load_model(
    'D:/Du_AN/NCKH/handwriting-siamese-one-shot-learning/siamese_model_backup_10_5_1.h5',
    custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
)

# Lấy toàn bộ test data
test_batch = test_data.batch(100)
test_input, test_val, y_true = next(iter(test_batch))
predictions = siamese_model.predict([test_input, test_val])

# Hiển thị kết quả
# Tạo lưới ảnh 5 hàng x 6 cột
plt.figure(figsize=(28, 25))

plt.figure(figsize=(12, 24))  # tăng chiều cao cho rõ

for i in range(20):
    anchor_img = test_input[i].numpy()
    val_img = test_val[i].numpy()
    pred = predictions[i][0]
    label = "🟢 Match" if pred > 0.5 else "🔴 Not Match"

    # Anchor
    plt.subplot(10, 4, 2*i + 1)
    plt.imshow(anchor_img)
    plt.title("Anchor")
    plt.axis('off')

    # Validation
    plt.subplot(10, 4, 2*i + 2)
    plt.imshow(val_img)
    plt.title(f"Pred: {pred:.2f}\n{label}")
    plt.axis('off')

# Điều chỉnh khoảng trắng — tránh che dòng đầu
plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95, bottom=0.03)
plt.show()
