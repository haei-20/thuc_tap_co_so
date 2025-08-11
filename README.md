
# Hệ thống nhận diện cảm xúc khuôn mặt

## Giới thiệu
Dự án web sử dụng Python để nhận diện cảm xúc khuôn mặt từ ảnh hoặc video webcam, ứng dụng các công nghệ học sâu và xử lý ảnh.

## Kiến trúc hệ thống
Hệ thống gồm các thành phần chính:

### 1. Giao diện Người dùng (Frontend)
- Phát triển bằng HTML, CSS, JavaScript (client-side), phục vụ qua Flask.
- Cho phép người dùng tải ảnh lên, bật/tắt camera, hiển thị video webcam và kết quả nhận diện (khuôn mặt, nhãn cảm xúc).
- Gửi các yêu cầu HTTP (POST ảnh, điều khiển camera, GET video) đến máy chủ backend.

### 2. Máy chủ Ứng dụng (Backend)
- Xây dựng bằng Flask (Python), xử lý logic và điều phối hoạt động hệ thống.
- Nhận yêu cầu từ frontend qua các API endpoint (ví dụ: /detect_emotion, /video_feed).
- Xử lý ảnh/video: phát hiện khuôn mặt, dự đoán cảm xúc bằng mô hình deep learning, trả kết quả về frontend.
- Quản lý trạng thái và điều khiển camera.

### 3. Xử lý Backend & Mô hình AI
- Phát hiện khuôn mặt bằng OpenCV và Haar Cascade.
- Tiền xử lý ảnh cho mô hình cảm xúc.
- Khởi tạo và vận hành mô hình nhận diện cảm xúc (VGG19), dự đoán cảm xúc.

## Chức năng cho Người dùng
- Nhận diện cảm xúc qua video trực tiếp: Bật/tắt camera, truyền hình ảnh webcam về server, phát hiện khuôn mặt và dự đoán cảm xúc, hiển thị nhãn cảm xúc trên video.
- Phân tích cảm xúc từ ảnh tĩnh: Tải ảnh hoặc chụp ảnh mới, gửi lên server để phân tích cảm xúc, hiển thị kết quả và biểu đồ xác suất cảm xúc.
- Điều chỉnh chế độ sáng/tối: Chuyển đổi giao diện sáng/tối linh hoạt.

## Công cụ & Thư viện sử dụng
- Visual Studio Code: Soạn thảo mã nguồn, quản lý phiên bản với Git.
- Flask: Framework web Python, xây dựng API endpoint.
- OpenCV (cv2): Xử lý ảnh/video, phát hiện khuôn mặt.
- NumPy: Xử lý mảng dữ liệu hình ảnh.
- TensorFlow/Keras: Xây dựng và sử dụng mô hình nhận diện cảm xúc (VGG19).
- Pillow (PIL): Đọc, ghi, chuyển đổi định dạng ảnh.
- Bootstrap 5, Chart.js, Font Awesome: Xây dựng giao diện web hiện đại, responsive, hiển thị biểu đồ và icon.
- JavaScript (ES6+): Logic phía client, gọi API, điều khiển giao diện động.

## Cấu trúc thư mục
```
├── app.py                        # File main chạy ứng dụng Flask
├── haarcascade_frontalface_default.xml # Bộ nhận diện khuôn mặt Haar Cascade
├── model.py                      # Định nghĩa mô hình nhận diện cảm xúc
├── model_weights.h5              # Trọng số mô hình đã huấn luyện
├── requirements.txt              # Danh sách các thư viện cần thiết
├── utils.py                      # Các hàm tiện ích
├── templates/
│   └── index.html                # Giao diện web
└── __pycache__/                  # File cache Python
```

## Hướng dẫn cài đặt & triển khai

### Backend
1. Khởi tạo thư mục dự án.
2. Tạo môi trường ảo Python:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
3. Cài đặt thư viện:
   ```powershell
   pip install -r requirements.txt
   ```
4. Chạy backend:
   ```powershell
   python app.py
   ```
   hoặc
   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 5000 --reload
   ```
   Truy cập: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Frontend
- Giao diện nằm trong file `templates/index.html`, gồm các thành phần: thanh điều hướng, video stream, biểu đồ cảm xúc, khu vực tải/chụp ảnh, hiển thị kết quả.

## Yêu cầu hệ thống
- Python 3.10+
- Các thư viện: Flask, OpenCV, TensorFlow/Keras, v.v. (xem `requirements.txt`)
