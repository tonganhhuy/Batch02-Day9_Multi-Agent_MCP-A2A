# Hướng Dẫn & Giới Thiệu Dự Án Nhóm: RAG Chatbot & Search Engine Ma Túy

Chào mừng bạn đến với thư mục **Lab_Assignment**. Đây là dự án nhóm hoàn chỉnh được phát triển dựa trên việc tích hợp các module RAG cá nhân từ Ngày 8, nâng cấp thành một hệ thống **Multi-Agent RAG Chatbot** tương tác, tích hợp kiểm định chất lượng tự động (Evaluation Pipeline).

---

## 📌 Tổng Quan Dự Án Nhóm

Dự án tập trung vào chủ đề: **Pháp luật Việt Nam về ma túy và các chất cấm** kết hợp với **tin tức liên quan đến nghệ sĩ và ma túy**. Nhóm đã xây dựng một giải pháp RAG toàn diện gồm hai phần chính:
1. **RAG Chatbot & Search Engine**: Ứng dụng Streamlit hỗ trợ hỏi đáp pháp lý có trích dẫn nguồn (citation), ghi nhớ ngữ cảnh hội thoại (memory), và hỗ trợ tìm kiếm kết hợp (Hybrid Search).
2. **RAG Evaluation Pipeline**: Hệ thống đánh giá tự động dựa trên thư viện DeepEval và Ragas để đo lường độ chính xác, độ tin cậy và sự liên quan của câu trả lời.

Toàn bộ mã nguồn cốt lõi của bài tập nhóm nằm trong thư mục [group_project](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project).

---

## 🛠️ Thành Phần & Cấu Trúc Mã Nguồn

Dưới đây là các tệp tin quan trọng trong bài tập nhóm:
* **Ứng dụng giao diện Chatbot**: [group_project/app.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/app.py) - Giao diện người dùng Streamlit hỗ trợ hội thoại, hiển thị tài liệu nguồn và điểm số liên quan.
* **Pipeline RAG Multi-Agent**: [group_project/multi_agent_rag.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/multi_agent_rag.py) - Định nghĩa kiến trúc đồ thị LangGraph điều phối các Agent chuyên biệt (Legal Agent, News Agent, Router, Aggregator).
* **Mô tả Kiến Trúc**: [group_project/MULTI_AGENT_ARCHITECTURE.md](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/MULTI_AGENT_ARCHITECTURE.md) - Tài liệu phân tích chi tiết sơ đồ luồng dữ liệu của hệ thống Multi-Agent RAG.
* **Mở rộng Server MCP**: [group_project/mcp_source_server.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/mcp_source_server.py) - Tích hợp giao thức Model Context Protocol để cung cấp nguồn dữ liệu động.
* **Pipeline Đánh Giá (Evaluation)**:
  * [group_project/evaluation/golden_dataset.json](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/evaluation/golden_dataset.json) - Tập dữ liệu gồm hơn 15 cặp câu hỏi-đáp mẫu chuẩn.
  * [group_project/evaluation/eval_pipeline.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/evaluation/eval_pipeline.py) - Script tự động chạy đánh giá hiệu năng RAG bằng các chỉ số Faithfulness, Answer Relevance, Context Recall, Context Precision.
  * [group_project/evaluation/results.md](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/evaluation/results.md) - Báo cáo kết quả kiểm thử chi tiết và phân tích các trường hợp kém chất lượng.

---

## 📐 Kiến Trúc Hệ Thống Multi-Agent RAG

Hệ thống hoạt động dựa trên đồ thị LangGraph được thiết kế như sau:

```
                  ┌──────────────────────┐
                  │      User Query      │
                  └──────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Query Router       │
                 └─────┬───────────┬─────┘
                       │           │
           (Legal Query)│           │(News/General Query)
                       ▼           ▼
        ┌──────────────────┐   ┌──────────────────┐
        │   Legal Agent    │   │    News Agent    │
        │  (ChromaDB RAG)  │   │  (ChromaDB RAG)  │
        └──────────┬───────┘   └───────────┬──────┘
                   │                       │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  Response Aggregator  │
                   └───────────┬───────────┘
                               │
                               ▼
                  ┌──────────────────────┐
                  │    Final Response    │
                  │   (with Citations)   │
                  └──────────────────────┘
```

### Chi tiết các bước xử lý:
1. **Thu thập & Chuẩn hóa**: Văn bản luật và báo chí thô từ `data/landing/` được chuyển đổi sang Markdown bằng Microsoft MarkItDown.
2. **Chunking & Indexing**: Chia nhỏ tài liệu và đánh chỉ mục vector vào ChromaDB sử dụng mô hình nhúng `BAAI/bge-m3`.
3. **Retrieval Pipeline**: Kết hợp tìm kiếm ngữ nghĩa (Semantic Search) và tìm kiếm từ khóa (BM25 Lexical Search), sau đó hợp nhất bằng RRF (Reciprocal Rank Fusion) và MMR Reranking.
4. **Multi-Agent Orchestration**: Router sẽ nhận diện ý định của người dùng để quyết định kích hoạt Legal Agent (luật ma túy) hay News Agent (tin tức liên quan nghệ sĩ). Aggregator sau đó tổng hợp thông tin, sắp xếp tài liệu nguồn tránh hiện tượng "lost-in-the-middle" và gửi LLM tạo phản hồi kèm trích dẫn nguồn `[Nguồn, Năm]`.

---

## 👥 Phân Công Công Việc Trong Nhóm

Dự án được hoàn thành với sự đóng góp tích cực từ các thành viên nhóm:

| Thành viên | MSSV | Vai Trò & Nhiệm Vụ Đảm Nhận | Trạng thái |
|---|---|---|---|
| **Mai Đức Vinh** | 2A202600587 | **Trưởng nhóm & Kỹ sư Data Pipeline**<br>Thu thập văn bản luật (Task 1), crawl tin tức nghệ sĩ (Task 2), chuẩn hóa Markdown (Task 3), quản lý Repo & tích hợp đồ thị Multi-Agent | ✅ Hoàn thành |
| **Tống Anh Huy** | 2A202600761 | **Kỹ sư Embedding & Vector Database**<br>Thực hiện Chunking, Indexing dữ liệu vào ChromaDB (Task 4) và viết module Semantic Search (Task 5). | ✅ Hoàn thành |
| **Nguyễn Mạnh Hiếu** | 2A202600887 | **Kỹ sư Retrieval & Reranking**<br>Xây dựng BM25 Lexical Search (Task 6), bộ Reranking RRF/MMR (Task 7) và tích hợp Hybrid Retrieval Pipeline (Task 9). | ✅ Hoàn thành |
| **Trần Duy Khánh** | 2A202600592 | **Kỹ sư LLM Generation & Frontend**<br>Tích hợp PageIndex Vectorless search (Task 8), xử lý chống "lost-in-middle" và sinh trích dẫn (Task 10), phát triển UI Chatbot bằng Streamlit. | ✅ Hoàn thành |
| **Nguyễn Đăng Khương** | 2A202600584 | **Kỹ sư Đánh Giá Chất Lượng (Evaluation)**<br>Thiết lập Golden Dataset 15+ mẫu, lập trình kiểm thử tự động với DeepEval/Ragas, thực hiện so sánh cấu hình A/B và viết báo cáo đánh giá kết quả. | ✅ Hoàn thành |

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.11+ và chạy các lệnh sau từ thư mục `Lab_Assignment`:

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Tạo tệp tin `.env` từ `.env.example` và bổ sung API keys của bạn:
```bash
cp .env.example .env
# Chỉnh sửa file .env để cấu hình OPENAI_API_KEY, GEMINI_API_KEY, v.v.
```

### 3. Chạy ứng dụng Chatbot
Khởi chạy giao diện Streamlit:
```bash
streamlit run group_project/app.py
```

### 4. Chạy kiểm tra đánh giá (Evaluation)
Để kiểm định hiệu năng RAG trên tập Golden Dataset:
```bash
python group_project/evaluation/eval_pipeline.py
```

*Để biết thêm thông tin chi tiết về từng yêu cầu làm bài tập nhóm ban đầu, vui lòng tham khảo tài liệu [group_project/README.md](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/Lab_Assignment/group_project/README.md).*