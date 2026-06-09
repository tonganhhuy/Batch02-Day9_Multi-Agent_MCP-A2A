# RAG Evaluation Results

## Framework sử dụng

> **DeepEval** (phiên bản 4.0.5) tích hợp bộ đo lường và so sánh cấu hình A/B RAG Pipeline.

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense Only - No Rerank) | Delta (Δ) |
|--------|---------------------------|----------------------------------|-----------|
| Faithfulness | 0.857 | 0.781 | 0.076 |
| Answer Relevance | 0.732 | 0.732 | 0.000 |
| Context Recall | 0.857 | 0.675 | 0.182 |
| Context Precision | 0.880 | 0.720 | 0.160 |
| **Average** | **0.831** | **0.727** | **0.104** |

---

## A/B Comparison Analysis

**Config A (Hybrid + Rerank):**
- Sử dụng tìm kiếm hỗn hợp: Semantic Search (ChromaDB Vector Store) kết hợp Lexical Search (BM25Okapi).
- Sử dụng **Reciprocal Rank Fusion (RRF)** để hợp nhất hai danh sách xếp hạng.
- Sử dụng mô hình **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`) để xếp hạng lại top 5 kết quả tối ưu.

**Config B (Dense Only - No Rerank):**
- Chỉ sử dụng Dense Search (Semantic Search) dựa trên khoảng cách Cosine.
- Không áp dụng xếp hạng lại (No Rerank).

**Kết luận:**
- **Config A vượt trội hơn Config B** trung bình **10.4%**.
- Việc kết hợp BM25 giúp tìm chính xác các điều luật có số hiệu cụ thể (ví dụ: "Điều 249"), trong khi Cross-Encoder giúp tinh lọc các đoạn văn bản có độ liên quan ngữ nghĩa cao nhất đưa lên đầu prompt, từ đó cải thiện đáng kể **Context Precision** và **Context Recall**.

---

## Worst Performers (Bottom 3)

Dựa trên kết quả chạy thử nghiệm với Config A, đây là 3 câu hỏi có điểm số thấp nhất:

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việt Nam gồm những chất nào? | 0.72 | 0.78 | 0.65 | Retrieval | Danh sách các chất ma tuý trong Nghị định 57 rất dài, việc chia nhỏ chunk 500 ký tự làm phân mảnh thông tin, dẫn tới recall kém. |
| 2 | Showbiz Việt phản ứng như thế nào trước tình trạng nghệ sĩ dính líu đến ma túy? | 0.75 | 0.80 | 0.70 | Generation | Câu trả lời của LLM mang tính chung chung, chưa liệt kê cụ thể các hành động tự nguyện xét nghiệm của nghệ sĩ. |
| 3 | Thời hạn cai nghiện ma túy bắt buộc đối với người từ đủ 18 tuổi trở lên theo Luật Phòng chống ma túy 2021 là bao lâu? | 0.80 | 0.82 | 0.72 | Retrieval | Các con số 12-24 tháng dễ bị lẫn lộn giữa cai nghiện bắt buộc và tự nguyện do overlap về mặt cấu trúc câu. |

---

## Recommendations

### Cải tiến 1: Tối ưu hóa Chunking Strategy cho Văn bản Pháp luật
**Action:** Sử dụng `MarkdownHeaderTextSplitter` hoặc gom nhóm theo Điều/Khoản thay vì chia đều `RecursiveCharacterTextSplitter` với 500 ký tự.  
**Expected impact:** Giữ trọn vẹn ngữ cảnh của từng điều luật, tăng Context Recall lên 10-15%.

### Cải tiến 2: Thêm Query Expansion (HyDE)
**Action:** Tạo tài liệu giả lập trước khi tìm kiếm ngữ nghĩa để chuyển đổi câu hỏi ngắn của người dùng thành cấu trúc giống văn bản luật.  
**Expected impact:** Tăng hiệu quả tìm kiếm ngữ nghĩa, cải thiện điểm số Answer Relevance.

### Cải tiến 3: Fine-tune Reranker hoặc dùng mô hình lớn hơn
**Action:** Chuyển sang sử dụng `Qwen/Qwen3-Reranker-0.6B` hoặc API của Jina Reranker v2 nếu điều kiện tài nguyên cho phép.  
**Expected impact:** Xếp hạng lại thông minh hơn, đẩy đúng ngữ cảnh hữu ích lên đầu, tăng Context Precision lên trên 0.92.
