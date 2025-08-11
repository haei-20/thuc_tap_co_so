import cv2

# Tắt OpenCL để tránh lỗi
cv2.ocl.setUseOpenCL(False)

# Dictionary ánh xạ số với tên cảm xúc
EMOTION_DICT = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


def detect_faces(frame):
    """
    Phát hiện khuôn mặt trong frame
    
    Args:
        frame: Frame ảnh cần phát hiện khuôn mặt
        
    Returns:
        tuple: (gray_frame, num_faces)
            - gray_frame: Frame ảnh grayscale
            - num_faces: Danh sách các khuôn mặt phát hiện được
    """
    # Chuyển đổi sang grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Sử dụng Haar Cascade để phát hiện khuôn mặt
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    num_faces = bounding_box.detectMultiScale(gray_frame)
    
    return gray_frame, num_faces

def draw_face_box(frame, face_coords, emotion_text):
    """
    Vẽ box xung quanh khuôn mặt và hiển thị cảm xúc
    
    Args:
        frame: Frame ảnh cần vẽ
        face_coords: Tọa độ khuôn mặt (x, y, w, h)
        emotion_text: Text cảm xúc cần hiển thị
    """
    x, y, w, h = face_coords
    # Vẽ rectangle xung quanh khuôn mặt
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    # Hiển thị text cảm xúc
    cv2.putText(frame, emotion_text, (x+20, y-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 