from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from model import create_emotion_model, predict_emotion
from utils import detect_faces, draw_face_box, EMOTION_DICT
import base64
import io
from PIL import Image
import traceback
from typing import Optional
from pydantic import BaseModel

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Khởi tạo camera và model
camera: Optional[cv2.VideoCapture] = None
# QUAN TRỌNG: Đảm bảo file 'model_weights.h5' nằm ở đúng vị trí
emotion_model = create_emotion_model(weights_path='model_weights.h5')

def get_camera():
    """Lấy camera object"""
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                camera.open(1)
            if not camera.isOpened():
                print("Không thể mở camera.")
                camera = None
        except Exception as e:
            print(f"Lỗi khi khởi tạo camera: {e}")
            camera = None
    return camera

@app.post("/camera/{action}")
async def camera_control(action: str):
    global camera
    try:
        if action == 'start':
            if camera is None:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    camera.open(1)
                if not camera.isOpened():
                    return {'status': 'error', 'message': 'Không thể khởi động camera'}
            elif not camera.isOpened():
                camera.open(0)
                if not camera.isOpened():
                    camera.open(1)
                if not camera.isOpened():
                    return {'status': 'error', 'message': 'Không thể mở lại camera'}
            return {'status': 'success', 'message': 'Camera đã bật'}
        
        elif action == 'stop':
            if camera is not None and camera.isOpened():
                camera.release()
            camera = None
            return {'status': 'success', 'message': 'Camera đã tắt'}
        
        return {'status': 'error', 'message': 'Invalid action'}
        
    except Exception as e:
        print(f"Lỗi trong camera control: {e}")
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}

def generate_frames():
    local_camera = get_camera()
    if local_camera is None or not local_camera.isOpened():
        print("Camera không sẵn sàng trong generate_frames.")
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    while True:
        if not local_camera.isOpened():
            print("Mất kết nối camera trong generate_frames.")
            break
        success, frame = local_camera.read()
        if not success:
            print("Không thể đọc frame từ camera.")
            break
        else:
            try:
                gray_frame, num_faces = detect_faces(frame)
                for face_coords in num_faces:
                    x, y, w, h = face_coords
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    if roi_gray_frame.size == 0:
                        continue
                    
                    maxindex, _ = predict_emotion(emotion_model, roi_gray_frame, return_probabilities=True)
                    emotion_text = EMOTION_DICT.get(maxindex, "Unknown")
                    draw_face_box(frame, face_coords, emotion_text)
            except Exception as e:
                print(f"Lỗi khi xử lý frame: {e}")
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), 
                           media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/detect_emotion")
async def detect_emotion(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_color is None:
            return {'error': 'Không thể đọc file ảnh'}
        
        gray_frame, num_faces = detect_faces(frame_color)
        
        if len(num_faces) == 0:
            return {'error': 'Không tìm thấy khuôn mặt nào trong ảnh'}
            
        x, y, w, h = num_faces[0]
        roi_gray_frame = gray_frame[y:y + h, x:x + w]

        if roi_gray_frame.size == 0:
            return {'error': 'Vùng khuôn mặt không hợp lệ'}
        
        maxindex, probabilities = predict_emotion(emotion_model, roi_gray_frame, return_probabilities=True)
        emotion_label = EMOTION_DICT.get(maxindex, "Unknown")
        
        return {
            'status': 'success',
            'emotion': emotion_label,
            'probabilities': probabilities.tolist()
        }

    except Exception as e:
        print("Đã xảy ra lỗi trong emotion detection:")
        traceback.print_exc()
        return {'error': f'Lỗi xử lý: {str(e)}'}

@app.get("/")
async def index():
    return FileResponse('templates/index.html')

if __name__ == '__main__':
    import uvicorn
    print("Starting server...")
    print("Server running at: http://127.0.0.1:5000")
    print("API documentation available at: http://127.0.0.1:5000/docs")
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
