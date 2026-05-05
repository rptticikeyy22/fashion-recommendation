# 👗 Multimodal Fashion Recommendation System
**Hệ thống Khuyến nghị và Tư vấn Thời trang Đa phương thức**

Dự án này là **Bài 2 (Đề xuất nhiệm vụ tự chọn)** thuộc Đồ án cuối kỳ môn Học Sâu[cite: 1]. 
Hệ thống giải quyết bài toán tư vấn và tìm kiếm thời trang thông minh bằng cách kết hợp sức mạnh của **Zero-shot Retrieval** và **Generative AI (Multimodal LLM)**.

---

## 🌟 Tính năng cốt lõi

1.  🔍 **Search by Text (Tìm kiếm bằng văn bản):** Người dùng nhập mô tả bằng ngôn ngữ tự nhiên (VD: "women white blazer"), hệ thống phân tích ngữ nghĩa và trả về các sản phẩm phù hợp.
2.  🖼️ **Search by Image (Tìm kiếm bằng hình ảnh):** Người dùng tải lên một hình ảnh tham khảo, hệ thống trích xuất đặc trưng hình ảnh và truy xuất các món đồ tương đồng về mặt thị giác.
3.  ✨ **AI Stylist (Tư vấn phối đồ thông minh):** Người dùng tải lên một món đồ (VD: một chiếc áo sơ mi). Hệ thống sử dụng mô hình LLM Đa phương thức để phân tích, đưa ra lời khuyên phối đồ (Mix & Match) bằng tiếng Việt, đồng thời tự động tạo chuỗi truy vấn (query) để tìm kiếm các món đồ (quần, giày, phụ kiện) phù hợp ngay trong cơ sở dữ liệu.

---

## ⚙️ Kiến trúc Hệ thống & Công nghệ

Hệ thống được xây dựng trên mô hình Micro-Architecture, bao gồm các công nghệ lõi:

*   **Image & Text Encoder:** Cấu trúc **OpenAI CLIP** (`clip-vit-base-patch32`). Mô hình này đóng vai trò nhúng (embedding) cả văn bản và hình ảnh vào cùng một không gian vector (Zero-shot).
*   **Vector Database:** **FAISS** (Facebook AI Similarity Search) được sử dụng để lập chỉ mục (indexing) và tìm kiếm hàng xóm gần nhất (K-Nearest Neighbors) với tốc độ siêu tốc trên hàng ngàn vector.
*   **LLM Reasoning:** Giao tiếp với API của **Google Gemini** (Các model: `gemini-1.5-flash`, `gemini-2.5-flash`, `gemini-3.1-pro`...) để thực hiện tác vụ nhận thức hình ảnh và sinh văn bản logic (AI Stylist).
*   **System Optimization:** 
    *   Tích hợp thuật toán **Load Balancing** (Cân bằng tải API Key) để luân phiên truy vấn, giải quyết bài toán Rate Limit (Error 429).
    *   Tích hợp **Streamlit Caching** để lưu trữ bộ nhớ đệm, tối ưu tốc độ phản hồi bằng 0 giây cho các truy vấn trùng lặp.
*   **Giao diện người dùng:** Xây dựng bằng framework **Streamlit** (Python).

---

## 📂 Kho Dữ liệu (Dataset & Checkpoints)

Theo yêu cầu của đồ án, toàn bộ dữ liệu và checkpoint đã được lưu trữ đầy đủ trên nền tảng đám mây[cite: 1]:

*   🤗 **Hugging Face Hub (Checkpoints):** [https://huggingface.co/rptticikeyy22/fashion-clip-faiss](https://huggingface.co/rptticikeyy22/fashion-clip-faiss) (Chứa file `fashion_clip.index` và `image_paths.npy`)
*   🤗 **Hugging Face Hub (Dataset):** [https://huggingface.co/datasets/rptticikeyy22/fashion-dataset-hd](https://huggingface.co/datasets/rptticikeyy22/fashion-dataset-hd) (Chứa tập dữ liệu hình ảnh `images_hd.zip`)

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Clone dự án
```bash
git clone [https://github.com/rptticikeyy22/fashion-recommendation.git](https://github.com/rptticikeyy22/fashion-recommendation.git)
cd fashion-recommendation
```

### Bước 2: Cài đặt môi trường
Đảm bảo bạn đang sử dụng Python 3.9 trở lên. Cài đặt các thư viện cần thiết thông qua file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 3: Tải dữ liệu Checkpoints & Dataset
Để chạy được ứng dụng, bạn cần tải lõi Vector Database và hình ảnh:
1. Truy cập vào link Hugging Face (Checkpoints) ở mục trên.
2. Tải 2 tệp: `fashion_clip.index` và `image_paths.npy`. Đặt 2 tệp này vào **thư mục gốc** của dự án.
3. Truy cập vào link Hugging Face (Dataset) ở mục trên.
4. Tải file nén `images_hd.zip`, giải nén ra thành thư mục `images_hd` và đặt vào thư mục gốc.

### Bước 4: Khởi chạy ứng dụng
Chạy lệnh sau trong Terminal (hoặc Command Prompt):
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở giao diện ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 📁 Cấu trúc Thư mục

```text
📦 Fashion_Recommendation_System
 ┣ 📂 images_hd/              # Thư mục chứa tập dữ liệu hình ảnh thời trang
 ┣ 📜 app.py                  # Mã nguồn chính (Giao diện, Load Balancing, gọi LLM)
 ┣ 📜 build_db.py             # Script trích xuất đặc trưng CLIP và xây dựng FAISS Index
 ┣ 📜 fashion_clip.index      # [Tải về] Vector Database (FAISS Index)
 ┣ 📜 image_paths.npy         # [Tải về] Mảng ánh xạ đường dẫn ảnh với Vector
 ┣ 📜 requirements.txt        # Danh sách thư viện phụ thuộc
 ┗ 📜 README.md               # Tài liệu dự án
```

---
*Dự án được phát triển nhằm mục đích nghiên cứu và học tập trong khuôn khổ học phần Học Sâu.*
