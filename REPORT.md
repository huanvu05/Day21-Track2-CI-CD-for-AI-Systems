# Báo Cáo Lab MLOps: CI/CD cho AI Systems

**Học viên:** Vũ Văn Huân
**Trường:** CMC University
**Khóa học:** AIInAction - VinUni (Day 21)
**Ngày thực hiện:** 07/05/2026

---

## 1. Lựa Chọn Siêu Tham Số (Hyperparameter Tuning)

Trong quá trình thực nghiệm cục bộ và theo dõi qua MLflow, tôi đã tiến hành tối ưu hóa mô hình `RandomForestClassifier`. 

**Bộ siêu tham số tối ưu được lựa chọn:**
- `n_estimators`: **150**
- `max_depth`: **15**
- `random_state`: **42**

**Lý do lựa chọn:**
Dựa trên kết quả từ MLflow UI, cấu hình này mang lại sự cân bằng tối ưu giữa độ chính xác (Accuracy) và khả năng tổng quát hóa. Việc tăng `n_estimators` lên 150 giúp mô hình ổn định hơn, trong khi giới hạn `max_depth` ở 15 ngăn chặn hiện tượng Overfitting, giúp mô hình hoạt động tốt trên cả tập Validation và Test.

---

## 2. So Sánh Hiệu Suất & Tự Động Hóa CI/CD

Hệ thống CI/CD được thiết lập để tự động kích hoạt quá trình huấn luyện lại khi có dữ liệu mới. Kết quả so sánh giữa hai giai đoạn cho thấy sức mạnh của việc mở rộng tập dữ liệu:

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) | Thay đổi |
|---|---|---|---|
| **Accuracy** | 0.672 | **0.764** | +9.2% |
| **F1_score** | 0.665 | **0.758** | +9.3% |

**Nhận xét:** Khi lượng dữ liệu tăng gấp đôi, hiệu suất mô hình cải thiện vượt bậc (vượt ngưỡng Eval Gate 0.70). Điều này khẳng định quy trình "Huấn luyện liên tục" (Continuous Training) hoạt động hiệu quả: Pipeline tự động nhận diện dữ liệu mới, huấn luyện, kiểm thử và sẵn sàng triển khai mà không cần can thiệp thủ công.

---

## 3. Nhật Ký Giải Quyết Sự Cố (Troubleshooting)

Trong quá trình triển khai hệ thống lên AWS EC2 thông qua GitHub Actions, tôi đã gặp và xử lý các vấn đề sau:

1. **Lỗi SSH Timeout (`dial tcp ***:22: i/o timeout`):**
   - **Nguyên nhân:** Địa chỉ IP Public của EC2 thay đổi sau khi khởi động lại, khiến GitHub Secrets (`VM_HOST`) bị lỗi thời.
   - **Giải quyết:** Cập nhật IP mới `13.214.121.107` vào GitHub Secrets và cấu hình lại Security Group để cho phép truy cập cổng 22 (SSH) và 8000 (API).

2. **Xung đột thư viện Cloud SDK:**
   - **Nguyên nhân:** File `serve.py` sử dụng `google.cloud.storage` trong khi môi trường lưu trữ hiện tại là AWS S3.
   - **Giải quyết:** Chuyển đổi mã nguồn sang sử dụng thư viện `boto3` và cập nhật `requirements.txt` để đảm bảo tính tương thích với hạ tầng AWS.