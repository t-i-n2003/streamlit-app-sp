
# 🧭 Roadmap 4 Tuần: Từ Dự Án Tốt → Sản Phẩm Gần Thực Chiến (Stock Prediction with LSTM)

## 🎯 MỤC TIÊU TỔNG:
Xây dựng một hệ thống dự đoán cổ phiếu:
- Xử lý nhiều mã tự động (batch pipeline),
- Đánh giá được hiệu quả mô hình bằng logic tài chính (backtest lãi/lỗ),
- Cải tiến mô hình theo hướng so sánh + tối ưu thực tế,
- Có giao diện demo trực quan,
- Trình bày logic và giải thích rõ ràng toàn bộ.

---

## 📅 ROADMAP 4 TUẦN

### 🔹 Tuần 1 – Dọn nền tảng: tổ chức lại pipeline + hiểu mô hình
**Mục tiêu:**
- Code gọn, modular.
- Mô hình hiểu từ gốc.
- Tách config, log rõ ràng.

**Việc cần làm:**
1. Tách quy trình thành 4 bước: `get_data(symbol)`, `clean_data(df)`, `train_model(df)`, `predict(df)`
2. Gộp model-saving/loading (`joblib`/`.save`) cho từng mã.
3. So sánh mô hình bạn đang dùng với:
   - Moving Average
   - ARIMA (`pmdarima.auto_arima`)
   - XGBoost Regressor (với `TimeSeriesSplit`)
4. Ghi lại mô hình nào đang tốt hơn theo MSE, MAE, MSLE.

**Bonus học thêm:**
- Đọc docs LSTM và ghi chú: `stateful`, `return_sequences`, `masking`,...

---

### 🔹 Tuần 2 – Đào sâu dự đoán nhiều bước + xử lý noise
**Mục tiêu:**
- Làm rõ multi-step forecasting.
- Giảm sai số chồng.

**Việc cần làm:**
1. Tách riêng logic dự đoán 1 bước vs nhiều bước (multi-step)
2. Vẽ biểu đồ dự đoán 1 bước + nhiều bước (3 ngày)
3. Xử lý lại nhiễu bằng:
   - Rolling mean (5 ngày)
   - Tạo thêm feature chênh lệch so với rolling

**Bonus học thêm:**
- Thử dùng Prophet để dự đoán định hướng seasonality

---

### 🔹 Tuần 3 – Gắn mô hình với logic đầu tư tài chính
**Mục tiêu:**
- Biến dự đoán thành chiến lược mua/bán giả định
- Đánh giá hiệu quả mô hình bằng logic sinh lời

**Việc cần làm:**
1. Tạo `simulate_trade(predictions, real_prices)`:
   - Dự đoán tăng ≥ 1% → mua, sau T+2 → tính lời/lỗ
2. Tính:
   - Tỉ suất lợi nhuận TB
   - Tỉ lệ lệnh lời/lỗ
   - Độ lệch chuẩn lợi nhuận
   - So sánh với random / buy-hold

**Bonus:**
- Viết báo cáo mô phỏng chiến lược đầu tư

---

### 🔹 Tuần 4 – Triển khai lại WebApp và ghi lại sản phẩm
**Mục tiêu:**
- Làm lại app gọn, có chọn mã cổ phiếu, số ngày dự đoán, biểu đồ
- Ghi lại toàn bộ như 1 product hoặc portfolio

**Việc cần làm:**
1. Giao diện streamlit:
   - Dropdown chọn mã
   - Biểu đồ train/test, mô phỏng lãi/lỗ
   - Cảnh báo bias model
2. Viết README + notebook:
   - Lý do chọn mô hình, xử lý dữ liệu, kết quả, điều chưa làm tốt

---

## 🎁 Bonus: Nếu bạn muốn thể hiện mức độ “sát thực tế” cao hơn
- Lưu log/predict vào SQLite hoặc Google Sheet
- Tự động chạy hàng ngày bằng cron + script
- Deploy lên HuggingFace Spaces hoặc Render
