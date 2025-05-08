# 📈 Dự đoán Giá Cổ Phiếu Trong Ngày Tiếp Theo Bằng LSTM
Dự án này sử dụng mô hình học sâu **Long Short-Term Memory (LSTM)** để dự đoán giá đóng cửa của cổ phiếu trong ngày tiếp theo dựa trên dữ liệu lịch sử.
## 📌 Mục tiêu
- Thu thập và tiền xử lý dữ liệu lịch sử giá cổ phiếu.
- Huấn luyện mô hình LSTM dựa trên chuỗi thời gian.
- Dự đoán giá đóng cửa của cổ phiếu trong ngày tiếp theo.
- Trực quan hóa kết quả dự đoán và so sánh với dữ liệu thực tế.
---
## 🛠️ Công nghệ sử dụng
- beautifulsoup4==4.13.4
- matplotlib==3.10.1
- numpy==1.26.4
- openpyxl==3.1.5
- pandas==2.2.0
- plotly==6.0.1
- protobuf==3.19.6
- scikeras==0.11.0
- scikit-learn
- scipy ==1.15.2
- streamlit==1.22.0
- tensorflow==2.11.0
- urllib3==2.4.0
- vnstock==3.2.4
---
## 📁 Cấu trúc thư mục
```
├── Models/             # File mô hình đã huấn luyện
├── Scalers/           # Scaler huấn luyện
├── clean_data.py       # Làm sạch dữ liệu
├── get_data.py         # Lấy dữ liệu thông qua API
├── predict_data.py    # Dữ đoán dữ liệu
├── train_model.py      # Huấn luyện mô hình bằng RandomizedSearchcV
├── train.py            # Huấn luyện cho 30 mã cổ phiếu
├── streamlit_stock.py  # triển khai mô hình bằng  Streamlit Cloud
├── README.md           # file hướng dẫn
├── requirements.txt
```
---
## 🚀 Cài đặt & chạy thử
### 1. Clone repository
```bash
git clone https://github.com/t-i-n2003/streamlit-app-sp
```
### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
### 3. Chạy huấn luyện mô hình
```bash
python train.py
```
### 4. Dự đoán giá ngày tiếp theo trong Streamlit
```bash
streamlit run streamlit_stock.py
```
## 📊 Kết quả
- Giá ngày hôm sau
---
## 📌 Ghi chú
- Dữ liệu có thể được lấy từ vnstock hoặc nguồn API khác.
- Mô hình có thể được mở rộng để dự đoán nhiều ngày hoặc thêm các chỉ báo kỹ thuật.
---
## 📬 Liên hệ
- 📧 Email: ntinit2003@gmail.com
- 🔗 LinkedIn: [Tin Nguyen](https://www.linkedin.com/in/tin-nguyen-04a86a278/)
