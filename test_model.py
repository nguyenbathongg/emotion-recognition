import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Các lớp cảm xúc
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Tải mô hình đã huấn luyện
def load_model_weights():
    # Tạo mô hình với kiến trúc tương tự như trong notebook
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        keras.layers.Conv2D(256, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(256, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(7, activation='softmax')
    ])
    
    # Tải trọng số đã huấn luyện
    model.load_weights('model_weights.h5')
    return model

# Xử lý ảnh đầu vào
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Chuyển sang ảnh grayscale
    img = img.resize((48, 48))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel
    img_array = img_array.reshape(1, 48, 48, 1)  # Thêm chiều batch và kênh
    return img_array

# Dự đoán cảm xúc
def predict_emotion(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    emotion = class_labels[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    
    # Hiển thị kết quả
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(class_labels, predictions[0])
    plt.title(f'Predicted: {emotion} ({confidence:.2f}%)')
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return emotion, confidence, predictions[0]

# Hàm main để chạy từ dòng lệnh
def main():
    if len(sys.argv) < 2:
        print("Sử dụng: python test_model.py <đường_dẫn_đến_ảnh>")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh tại: {image_path}")
        return
    
    print(f"Đang tải mô hình...")
    model = load_model_weights()
    
    print(f"Đang dự đoán cảm xúc cho ảnh: {image_path}")
    emotion, confidence, _ = predict_emotion(model, image_path)
    
    print(f"\nKết quả dự đoán: {emotion}")
    print(f"Độ tin cậy: {confidence:.2f}%")

if __name__ == "__main__":
    main()
