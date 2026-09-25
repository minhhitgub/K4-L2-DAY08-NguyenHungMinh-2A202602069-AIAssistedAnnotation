# Nguồn ngoài và tình trạng phát hành

**Bản ghép thử nội bộ.** Các frame trong `data/` được trích từ video YouTube
[`Ina7KMV2OEI`](https://www.youtube.com/watch?v=Ina7KMV2OEI) theo
[`data/DATA.md`](data/DATA.md). Chưa có bằng chứng trong hai repo rằng chủ sở hữu video đã cho phép
phân phối lại frame qua repo học viên public. Người phụ trách dữ liệu/VinUni cần xác nhận trước
khi công bố. Tài liệu này không tự cấp quyền sử dụng.

Notebook sử dụng [Ultralytics](https://github.com/ultralytics/ultralytics) bản `8.4.161` trên Colab;
điều khoản phần mềm và mô hình áp dụng theo nhà phát hành. Bài lab dùng [CVAT Community](https://github.com/cvat-ai/cvat)
`v2.76.0` chạy qua Docker trên máy của từng học viên; CVAT không được đóng gói trong repo.

Nhãn test là đầu ra mô hình, chưa qua rà nhãn thủ công. Vai trò và giới hạn của chúng được ghi
trong [`data/DATA.md`](data/DATA.md).
