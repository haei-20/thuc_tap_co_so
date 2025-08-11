 Mô hình kiến trúc hệ thống 
Hệ thống nhận diện cảm xúc khuôn mặt được thiết kế với các thành phần chính sau 
• Giao diện Người dùng (Frontend và JavaScript phía client): 
o Là giao diện web chính, được phát triển bằng HTML, CSS và 
JavaScript, được phục vụ bởi FastAPI. 
o Tương tác trực tiếp với người dùng: cho phép tải ảnh lên, điều khiển 
(bật/tắt) camera. 
o Hiển thị luồng video từ webcam và các kết quả nhận diện (khuôn mặt 
được khoanh vùng, nhãn cảm xúc dự đoán). 
o Gửi các yêu cầu HTTP (ví dụ: POST để tải ảnh, POST để điều khiển 
camera, GET để nhận luồng video) đến Máy chủ Ứng dụng (FastAPI) 
• Máy chủ Ứng dụng (Server – FastAPI) 
o Là trung tâm xử lý logic và điều phối các hoạt động của hệ thống, được 
xây dựng bằng FastAPI (Python). 
o Tiếp nhận các yêu cầu HTTP từ Giao diện Người dùng qua các RESTful 
API (ví dụ: /detect_emotion cho ảnh tải lên, /video_feed cho luồng 
video, /camera/{action} để quản lý camera). 
o Thực hiện xử lý logic nghiệp vụ: 
▪ Đối với ảnh tải lên: gọi module xử lý ảnh để phát hiện khuôn 
mặt, sau đó gọi module mô hình để dự đoán cảm xúc và trả kết 
quả về cho frontend. 
▪ Đối với luồng video: liên tục lấy khung hình từ camera, xử lý 
từng khung hình để phát hiện khuôn mặt, dự đoán cảm xúc, vẽ 
kết quả lên khung hình và truyền (stream) về frontend. 
▪ Quản lý trạng thái và điều khiển thiết bị camera. 
• Xử lý Backend và mô hình: bao gồm phát hiện khuôn mặt, tiền xử lý ảnh cho 
mô hình cảm xúc, khởi tạo và vận hành mô hình Nhận diện cảm xúc và dự 
đoán cảm xúc


Chức năng cho Người dùng - - - 
Nhận diện cảm xúc qua video trực tiếp: Hệ thống cho phép người dùng 
bật/tắt camera trực tiếp trên trình duyệt. Khi camera được bật, hình ảnh từ 
webcam sẽ được truyền liên tục về server. Server sử dụng thư viện OpenCV 
để phát hiện khuôn mặt trong từng khung hình, sau đó cắt vùng khuôn mặt và 
đưa vào mô hình deep learning (VGG19 đã huấn luyện) để dự đoán cảm xúc. 
Kết quả nhận diện sẽ được hiển thị trực tiếp trên video với nhãn cảm xúc 
Phân tích cảm xúc từ ảnh tĩnh:  
+  Tải ảnh từ thiết bị 
+  Chụp ảnh mới 
Ảnh này sẽ được gửi lên server, xử lý tương tự như video stream: phát hiện 
khuôn mặt, phân tích cảm xúc và trả về kết quả. Giao diện web hiển thị ảnh đã 
chọn, kết quả nhận diện và biểu đồ xác suất cảm xúc. Có thể: 
Điều chỉnh chế độ sáng/tối của giao diện: chuyển đổi giữa giao diện sáng/tối 
linh hoạt với môi trường

I. CÀI ĐẶT HỆ THỐNG VÀ TRIỂN KHAI HỆ THỐNG 
1.  Một số công cụ sử dụng. 
• Visual Studio Code (VS Code): 
○ Trình soạn thảo mã nguồn phổ biến với nhiều tiện ích mở rộng hỗ 
trợ phát triển dự án. 
○ Tích hợp Git giúp dễ dàng quản lý và kiểm soát phiên bản mã nguồn. 
2.  Thư viện hỗ trợ. 
• FastAPI: Framework web Python hiện đại, hiệu suất cao, dùng để xây dựng 
các API endpoint cho hệ thống. 
• Uvicorn: ASGI server dùng để chạy ứng dụng FastAPI. 
• OpenCV (cv2): Thư viện xử lý ảnh và video, dùng để phát hiện khuôn mặt 
và xử lý frame từ camera. 
• NumPy: Thư viện tính toán số học, xử lý mảng dữ liệu hình ảnh. 
• TensorFlow/Keras: Framework học sâu, dùng để xây dựng và sử dụng mô 
hình nhận diện cảm xúc (VGG19). 
• Pillow (PIL): Thư viện xử lý ảnh hỗ trợ đọc, ghi và chuyển đổi định dạng 
ảnh. 
• Base64, io: Hỗ trợ mã hóa/giải mã và xử lý dữ liệu nhị phân. 
• Pydantic: Hỗ trợ kiểm tra và xác thực dữ liệu đầu vào cho FastAPI. 
• Bootstrap 5: Framework CSS giúp xây dựng giao diện web hiện đại, 
responsive. 
• Chart.js: Thư viện JavaScript để hiển thị biểu đồ xác suất cảm xúc trên giao 
diện web. 
• Font Awesome: Thư viện icon vector cho giao diện người dùng. 
• JavaScript (ES6+): Xử lý logic phía client, gọi API, điều khiển giao diện 
động. 
3. Cài đặt và triển khai phía Backend 
• Chuẩn bị môi trường:  
o Bước 1: Khởi tạo thư mục dự án 
Đặng Thị Hà - B22DCCN253 
Nguyễn Thị Hiền - B22DCCN289      
35 
o Bước 2: Tạo môi trường ảo python venv bằng câu lệnh [python -m venv 
venv] trên Windows và [source venv/bin/activate] trên Linux/macOS 
o Bước 3: Kích hoạt môi trường ảo python venv bằng câu lệnh 
[venv\Scripts\activate] với window 
o Bước 4: Đặt các thư viện cần thiết cho dự án trong file requirements.txt 
o Bước 5: Cài đặt các thư viện cần thiết bằng câu lệnh [pip install -r 
requirements.txt] 
• Triển khai Backend: 
o Backend của hệ thống được triển khai thông qua ba module chính: 
▪ app.py - File chính định nghĩa các API endpoint và xử lý các 
request từ client. Các chức năng chính bao gồm: 
• Khởi tạo ứng dụng FastAPI và mô hình AI 
• Quản lý kết nối camera  
• Cung cấp video stream xử lý theo thời gian thực  
• Xử lý tải lên ảnh và phân tích cảm xúc 
• Phục vụ giao diện web 
▪ model.py - Mô hình nhận diện cảm xúc, chịu trách nhiệm: 
• Tạo mô hình nhận diện cảm xúc dựa trên VGG19 
• Tải trọng số đã huấn luyện 
• Thực hiện dự đoán cảm xúc từ ảnh khuôn mặt 
▪ utils.py - Xử lý hình ảnh, cung cấp thông tin về : 
• Hằng số và ánh xạ tên cảm xúc 
• Hàm phát hiện khuôn mặt sử dụng Haar Cascade 
• Hàm vẽ khung và nhãn cảm xúc lên hình ảnh 
• Khởi chạy backend: 
o Để chạy backend sử dụng lệnh [python app.py] hoặc sử dụng Uvicorn 
trực tiếp: [uvicorn app:app --host 127.0.0.1 --port 5000 –reload] 
o Server sẽ chạy tại địa chỉ http://127.0.0.1:5000 với tài liệu API tự động 
sinh tại http://127.0.0.1:5000/docs. 
4. Cài đặt và triển khai phía FE 
• Cấu trúc giao diện:  Giao diện được chia thành các thành phần chính: 
o Thanh điều hướng với logo và công tắc chuyển đổi theme 
o Phần hiển thị video stream và điều khiển camera 
o Biểu đồ xác suất cảm xúc 
o Khu vực tải lên ảnh và chụp ảnh mới 
o Phần hiển thị kết quả nhận diện 
Đặng Thị Hà - B22DCCN253 
Nguyễn Thị Hiền - B22DCCN289      
36 
• Triển khai Frontend: File index.html trong thư mục templates chứa toàn bộ 
giao diện người dùng, bao gồm HTML, CSS và JavaScript. Dưới đây là các 
thành phần chính: 
