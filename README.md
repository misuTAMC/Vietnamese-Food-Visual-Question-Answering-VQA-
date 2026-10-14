# Vietnamese Food Visual Question Answering (VQA)

Dự án fine-tune mô hình **Qwen2-VL-2B-Instruct** bằng LoRA để xây dựng hệ thống hỏi đáp hình ảnh món ăn Việt Nam. Dự án tập trung vào pipeline huấn luyện đa phương thức, debugging lỗi label masking và đánh giá chất lượng câu trả lời bằng nhiều chỉ số khác nhau.

## 1. Tổng quan

Visual Question Answering (VQA) là bài toán kết hợp thông tin từ hình ảnh và câu hỏi bằng ngôn ngữ tự nhiên để sinh câu trả lời phù hợp.

Trong dự án này, mô hình nhận hình ảnh món ăn và câu hỏi bằng tiếng Việt, sau đó tạo câu trả lời dưới dạng văn bản tự do.

## 2. Công nghệ và dữ liệu

| Thành phần | Chi tiết |
|---|---|
| Mô hình nền | Qwen2-VL-2B-Instruct |
| Phương pháp fine-tuning | LoRA |
| Framework | PyTorch |
| Thư viện | Transformers, PEFT, qwen-vl-utils |
| Dataset | ViFoodVQA |
| Số cặp hỏi–đáp | 23.260 |
| Số hình ảnh | 2.326 |
| Số nhóm món ăn | 103 |
| Phần cứng huấn luyện | 1 NVIDIA RTX 3090 |

### Cấu hình LoRA

- Rank (`r`): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: toàn bộ linear layers
- Số tham số có thể huấn luyện: khoảng 29 triệu, theo kết quả thống kê của lần huấn luyện.

## 3. Pipeline huấn luyện

Pipeline được triển khai bằng PyTorch, bao gồm:

1. Nạp dữ liệu hình ảnh và các cặp hỏi–đáp từ JSONL.
2. Tiền xử lý hình ảnh và văn bản bằng processor của Qwen2-VL.
3. Xây dựng custom dataset và data collator cho đầu vào đa phương thức.
4. Mask prompt và padding bằng `-100` để loss chỉ được tính trên các token câu trả lời mục tiêu.
5. Fine-tune mô hình bằng LoRA.
6. Tối ưu hóa với AdamW, gradient accumulation, gradient clipping và cosine learning-rate scheduler.
7. Theo dõi validation loss, lưu checkpoint và áp dụng early stopping.

## 4. Điểm nổi bật: Debugging lỗi label masking

Trong quá trình phát triển, mô hình sau fine-tuning ban đầu đạt kết quả đánh giá giống hệt mô hình nền, mặc dù quá trình huấn luyện vẫn chạy bình thường.

Qua kiểm tra, vấn đề được xác định nằm ở cách căn chỉnh nhãn trong data collator. Cơ chế xử lý đầu vào đa phương thức của Qwen2-VL, bao gồm việc mở rộng token hình ảnh động, khiến ranh giới giữa prompt và câu trả lời cần được xác định chính xác.

**Cách khắc phục:** Điều chỉnh logic masking bằng cách xác định ranh giới lượt trả lời của assistant trực tiếp trong token IDs, sau đó mask phần prompt trước ranh giới này.

Việc sửa lỗi giúp loss được tính trên các token câu trả lời mục tiêu, thay vì vô tình giám sát sai phần đầu vào.

Bài học rút ra: training loss giảm không đồng nghĩa với việc mô hình đã học đúng mục tiêu. Cần kiểm tra dữ liệu, nhãn giám sát và kết quả evaluation để xác nhận pipeline hoạt động đúng.

## 5. Phương pháp đánh giá

Dự án kết hợp các chỉ số đo độ tương đồng từ vựng và độ tương đồng ngữ nghĩa.

### Thiết kế tập đánh giá

- Sử dụng Gemini API để phân loại các cặp hỏi–đáp thành 5 nhóm câu hỏi.
- Thiết kế tập đánh giá phân tầng theo 103 nhóm món ăn.
- Mỗi nhóm món ăn có 15 cặp hỏi–đáp, tổng cộng 1.545 mẫu đánh giá.
- So sánh mô hình nền và mô hình sau fine-tuning.

### Các chỉ số

- **BLEU:** Đo mức độ trùng khớp n-gram giữa câu trả lời dự đoán và câu tham chiếu.
- **ROUGE-L:** Đo độ tương đồng dựa trên dãy con chung dài nhất.
- **METEOR:** Đánh giá độ tương đồng từ vựng và căn chỉnh từ.
- **BERTScore:** Đo độ tương đồng ngữ nghĩa bằng biểu diễn ngữ cảnh từ PhoBERT.
- **Exact Match:** Đo tỷ lệ câu trả lời khớp chính xác với câu tham chiếu.

## 6. Kết quả thực nghiệm

| Chỉ số | Trước fine-tuning | Sau fine-tuning |
|---|---:|---:|
| BLEU | 7.79 | 31.94 |
| ROUGE-L | 44.75 | 61.72 |
| METEOR | 34.53 | 52.39 |
| BERTScore | 53.27 | 72.20 |
| Exact Match | 0.0 | 7.9 |

Các giá trị được báo cáo theo thang điểm của lần đánh giá trong dự án.

Mô hình sau fine-tuning cải thiện trên tất cả các chỉ số được báo cáo. Exact Match vẫn thấp do câu trả lời được sinh dưới dạng văn bản tự do, nên nội dung tương đương về ý nghĩa không nhất thiết khớp chính xác với câu tham chiếu.

Vì vậy, ROUGE-L và BERTScore được sử dụng bổ sung để đánh giá chất lượng câu trả lời ngoài mức độ trùng khớp chính xác.



## 9. Hạn chế và hướng phát triển

- Phân tích lỗi theo từng nhóm câu hỏi và nhóm món ăn.
- Đánh giá khả năng tổng quát hóa trên các nhóm món ăn chưa xuất hiện trong tập huấn luyện.
- Bổ sung ví dụ dự đoán và biểu đồ train/validation loss.
- Tài liệu hóa đầy đủ workflow evaluation và cấu hình metric để tăng khả năng tái lập kết quả.
