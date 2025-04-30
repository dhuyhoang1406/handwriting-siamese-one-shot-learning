# SCHEDULE.md

## Tên đề tài

**Nhận dạng chữ viết tay một lần bằng Mạng Siamese Neural**

## Thành viên / Thông tin liên hệ

- **Đặng Huy Hoàng (3122560019)** - Email: dhuyhoang181@gmail.com - SĐT: 0898840017
- **Đặng Huy Hoàng (3122560020)** - Email: Huyhoang119763@gmail.com - SĐT: 0585822397

## Kế hoạch dự kiến

1. **Nghiên cứu, chọn đề tài, tạo sản phẩm, xây dựng đề cương** (Tuần 1-2)
2. **Nộp đề cương, sửa chữa và hoàn thiện đề cương** (Tuần 3)
3. **Nghiên cứu, viết và hoàn thiện luận văn** (Tuần 4-12)
   - **Chương 1: Tổng quan vấn đề**
   - **Chương 2: Lược khảo tài liệu (Literature Review)**
   - **Chương 3: Phương pháp nghiên cứu (Methodology)**
   - **Chương 4: Thực nghiệm và Thảo luận**
   - **Chương 5: Kết luận và Hướng phát triển**
4. **Chỉnh sửa, hoàn thiện và bảo vệ luận văn** (Tuần 13-14)

## Liên kết

- [Kế hoạch chi tiết](./schedule.xlsx)

## Tổ chức công việc nhóm

- **Phân công nhiệm vụ**:
  - **Tuần 1-3**: Phân tích bài toán nghiên cứu (Nhóm trưởng):
    - Xác định yêu cầu bài toán (Tập dữ liệu Omniglot, Input: cặp hình ảnh, Output: xác suất giống/khác).
    - Đánh giá hiện trạng và các thách thức trong nhận dạng chữ viết tay một lần.
  - **Tuần 4-5**: Tìm hiểu và nghiên cứu các mô hình hiện tại (Nhóm trưởng):
    - Khảo sát các phương pháp CNN và Siamese Neural Network cho one-shot learning.
    - Triển khai mô hình Siamese cơ sở dựa trên bài báo của Koch et al.
    - Đánh giá ưu, nhược điểm của các phương pháp SNN và các mô hình cơ sở (HBPL, 1-NN).
  - **Tuần 6-8**: Thực nghiệm các mô hình hiện có (Tất cả thành viên):
    - Huấn luyện và đánh giá mô hình trên tập Omniglot (30k, 90k, 150k cặp).
    - Áp dụng các kỹ thuật tối ưu hóa (SGD với momentum, điều chuẩn L2).
    - Đánh giá kết quả bằng các độ đo: Accuracy, Precision, Recall, F1-score.
  - **Tuần 9-10**: Phát biểu bài toán nghiên cứu và đề xuất cải tiến (Tất cả thành viên):
    - Xây dựng bài toán nghiên cứu từ các hạn chế đã phân tích (ví dụ: độ bền với biến đổi nét vẽ).
    - Đề xuất cải tiến như tăng cường dữ liệu affine hoặc biến đổi nét vẽ (xem Hình 8 trong bài báo).
  - **Tuần 11**: Phát triển thuật toán và thực nghiệm mô hình cải tiến (Tất cả thành viên):
    - Thiết kế kiến trúc mô hình cải tiến (ví dụ: thêm biến đổi nét vẽ, điều chỉnh số tầng CNN).
    - Huấn luyện và kiểm thử mô hình cải tiến trên Omniglot, thử nghiệm trên MNIST để đánh giá tổng quát hóa.
    - Tăng cường dữ liệu bằng các kỹ thuật như biến đổi affine (xoay, cắt, tỷ lệ, dịch chuyển).
  - **Tuần 12**: Phân tích và đánh giá kết quả nghiên cứu (Tất cả thành viên):
    - So sánh mô hình cải tiến với mô hình cơ sở và các phương pháp khác (HBPL, 1-NN).
    - Đánh giá bằng các độ đo: Accuracy, Precision, Recall, F1-score.
  - **Tuần 13**: Viết báo cáo tổng kết và chuẩn bị nghiệm thu (Tất cả thành viên):
    - Soạn thảo báo cáo tổng kết nghiên cứu.
    - Chuẩn bị tài liệu và các kết quả minh họa (biểu đồ, hình ảnh kết quả) phục vụ nghiệm thu.

- **Họp nhóm định kỳ**: 2 lần/tuần vào thứ Ba và thứ Sáu để cập nhật tiến độ.
- **Công cụ quản lý**:
  - **GitHub**: Quản lý mã nguồn và tài liệu.
  - **Google Drive**: Lưu trữ dữ liệu và tài liệu nhóm.
- **Báo cáo tiến độ**: Cập nhật tiến độ vào mỗi cuối tuần, điều chỉnh kế hoạch nếu cần.

---