import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19


def create_emotion_model(weights_path='model_weights.h5'):
    """
    Tạo và trả về mô hình VGG19 để nhận diện cảm xúc

    """
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    x = base_model.layers[-2].output 
    x = GlobalAveragePooling2D()(x)
    num_classes = 7
    predictions = Dense(num_classes, activation='softmax', name='out_layer')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    try:
        model.load_weights(weights_path)
        print(f"Đã tải trọng số từ {weights_path} thành công.")
    except Exception as e:
        print(f"Lỗi khi tải trọng số từ {weights_path}: {e}")
        print("Mô hình sẽ sử dụng trọng số khởi tạo (ImageNet cho VGG19 base, ngẫu nhiên cho lớp Dense mới).")
        
    return model

def predict_emotion(model, face_img, return_probabilities=False):
    """
    Dự đoán cảm xúc từ ảnh khuôn mặt (ảnh xám).
    Ảnh sẽ được tiền xử lý để phù hợp với mô hình VGG19.
    Trả về chỉ số của cảm xúc dự đoán và (tùy chọn) mảng xác suất.
    """
    processed_img = cv2.resize(face_img, (48, 48))
    
    if len(processed_img.shape) == 2 or processed_img.shape[2] == 1:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    
    processed_img = np.expand_dims(processed_img, axis=0)
    processed_img = processed_img / 255.0

    predictions = model.predict(processed_img)
    maxindex = int(np.argmax(predictions[0])) # Chuyển sang int để jsonify
    
    if return_probabilities:
        return maxindex, predictions[0] 
    return maxindex
