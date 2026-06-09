# Báo Cáo Kết Quả Hoàn Thành Codelab (Lab-Solution.md)

Hệ thống Multi-Agent hỗ trợ phân tích pháp lý đã được hoàn thiện đầy đủ từ **Stage 1** đến **Stage 5** cùng các bài tập bổ trợ trong tệp [CODELAB.md](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/CODELAB.md). Dưới đây là chi tiết các thay đổi đã thực hiện trên codebase.

---

## 1. Phần 1: Direct LLM Calling (Stage 1)

### Các thay đổi chính:
- **Tối ưu hóa cấu hình LLM**: Cập nhật hàm [get_llm](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/llm.py#L29-L39) trong [common/llm.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/llm.py):
  - Thiết lập `temperature=0.3` để giảm tính ngẫu nhiên, tăng độ chính xác và ổn định cho phản hồi pháp lý.
  - Nâng `max_tokens` từ `300` lên `800` để đảm bảo LLM không bị cắt cụt câu trả lời khi giải thích các khía cạnh pháp lý phức tạp.
  - Cập nhật thư viện khởi tạo `ChatOpenAI` để tương thích tốt hơn: thay đổi `openai_api_base` thành `base_url` và `openai_api_key` thành `api_key`.
- **Thêm tính năng tự động validate**:
  - Viết hàm [validate_openrouter_config](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/llm.py#L11-L26) kiểm tra tính hợp lệ của `OPENROUTER_API_KEY` (phải bắt đầu bằng `sk-or-`) và `OPENROUTER_MODEL` ngay tại thời điểm khởi động hệ thống. Nếu cấu hình sai hoặc thiếu, chương trình sẽ ném ra lỗi `RuntimeError` chi tiết để người dùng sửa đổi tệp `.env`.

---

## 2. Phần 2: LLM + RAG & Tools (Stage 2)

### Các thay đổi chính:
- **Mở rộng Cơ sở kiến thức (Knowledge Base)**:
  - Thêm thông tin liên quan đến **Luật Lao động Việt Nam 2019** vào danh sách `LEGAL_KNOWLEDGE` trong [stages/stage_2_rag_tools/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_2_rag_tools/main.py) với các từ khóa tìm kiếm tiếng Việt và tiếng Anh: `["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"]`.
- **Tạo và tích hợp công cụ tính thời hiệu khởi kiện**:
  - Định nghĩa công cụ `@tool` [check_statute_of_limitations](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_2_rag_tools/main.py#L148-L161) để kiểm tra thời hiệu khởi kiện của các loại vụ án (`contract` - 4 năm theo UCC § 2-725, `tort` - 2-3 năm, `property` - 5 năm).
  - Tích hợp công cụ này vào danh sách `TOOLS` và bổ sung xử lý trong vòng lặp gọi công cụ thủ công ở cả [stages/stage_2_rag_tools/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_2_rag_tools/main.py) và tệp bài tập [exercises/exercise_2_tools.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/exercises/exercise_2_tools.py).

---

## 3. Phần 3: Single Agent với ReAct (Stage 3)

### Các thay đổi chính:
- **Thêm công cụ tra cứu án lệ**:
  - Viết `@tool` [search_case_law](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_3_single_agent/main.py#L175-L192) trong [stages/stage_3_single_agent/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_3_single_agent/main.py) hỗ trợ tra cứu các án lệ nổi tiếng như *Hadley v. Baxendale*, *Donoghue v. Stevenson*, *Carlill v. Carbolic Smoke Ball Co*.
- **Debug Reasoning**:
  - Thêm tham số `debug=True` vào hàm `create_react_agent` để hiển thị tường tận luồng suy nghĩ của Agent (Think -> Act -> Observe) trực tiếp trên terminal trong quá trình chạy.

---

## 4. Phần 4: Multi-Agent In-Process (Stage 4)

### Các thay đổi chính:
- **Xây dựng Privacy Agent chuyên biệt**:
  - Định nghĩa node [privacy_agent](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/exercises/exercise_4_multiagent.py#L95-L107) / [call_privacy_specialist](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_4_milti_agent/main.py#L303-L325) xử lý khía cạnh GDPR và bảo mật thông tin cá nhân trong [stages/stage_4_milti_agent/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_4_milti_agent/main.py) và [exercises/exercise_4_multiagent.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/exercises/exercise_4_multiagent.py).
  - Khai báo các trạng thái bổ sung `needs_privacy` và `privacy_result` trong shared state `State`/`LegalState`.
- **Cập nhật Điều hướng Đồ thị (Routing logic)**:
  - Cải tiến node [check_routing](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_4_milti_agent/main.py#L150-L205) để phát hiện tự động các từ khóa liên quan đến bảo mật dữ liệu (`data`, `privacy`, `gdpr`, `dữ liệu`) trong câu hỏi gốc nhằm điều phối yêu cầu sang Privacy Agent.
  - Đăng ký node mới vào `StateGraph` và kết nối luồng xử lý từ `check_routing` -> `privacy_agent` -> `aggregate_results`.
- **Tổng hợp kết quả**:
  - Cập nhật hàm [aggregate](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_4_milti_agent/main.py#L274-L300) để đính kèm kết quả phân tích bảo mật dữ liệu của Privacy Agent vào báo cáo pháp lý cuối cùng gửi cho người dùng.
- **Trực quan hóa đồ thị & Demos**:
  - Nhúng thư viện đồ họa của IPython hiển thị sơ đồ Mermaid PNG trực quan.
  - Tạo tệp giao diện tương tác [docs/stage_4_multi_agent_interactive_demo.html](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/docs/stage_4_multi_agent_interactive_demo.html).

---

## 5. Phần 5: Distributed A2A System (Stage 5)

Đây là phần chứa nhiều tối ưu hóa hiệu năng chuyên sâu cùng việc cấu hình chạy phân tán trên Windows:

### A. Tối ưu hóa hiệu năng truyền thông & Client
- **Chia sẻ kết nối HTTP (Client Caching)**:
  - Thay vì khởi tạo một `httpx.AsyncClient` mới trên mỗi lượt gọi như bản gốc, hệ thống đã được cập nhật thêm hàm `get_client()` trong cả [common/a2a_client.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/a2a_client.py) và [common/registry_client.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/registry_client.py) để tái sử dụng một client duy nhất, giảm thiểu đáng kể overhead bắt tay TCP/TLS.
- **Bộ nhớ đệm thông tin cấu hình (Cache metadata)**:
  - Áp dụng `_card_cache` lưu trữ cấu hình `AgentCard` thu thập được và `_discover_cache` lưu trữ ánh xạ nhiệm vụ-địa chỉ từ registry. Nhờ vậy, các Agent không cần liên tục truy vấn Registry hoặc tải lại thông tin thẻ đại lý trên mỗi request phân phối, giúp giảm độ trễ (RTT) cực lớn.

### B. Tối ưu hóa mạng lưới Agent & Giảm số lượt gọi LLM
- **Bỏ vòng lặp ReAct thừa (Direct LLM Node)**:
  - Cập nhật [tax_agent/graph.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/tax_agent/graph.py) và [compliance_agent/graph.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/compliance_agent/graph.py) bằng việc chuyển đổi cấu trúc đồ thị từ `create_react_agent` sang một `StateGraph` cơ bản với 1 node `call_llm` duy nhất. Việc này loại bỏ overhead của mô hình suy nghĩ lặp ReAct không cần thiết do hai Agent này không sử dụng công cụ nào bên ngoài.
- **Gộp Node để giảm truy vấn LLM**:
  - Tại [law_agent/graph.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/law_agent/graph.py), tiến hành gộp node phân tích pháp lý ban đầu (`analyze_law`) và node điều phối (`check_routing`) thành một node duy nhất có tên [analyze_and_route](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/law_agent/graph.py#L54-L121). Node này gọi LLM đúng 1 lần duy nhất để trả về kết quả dạng JSON chứa cả phần phân tích tổng quan lẫn quyết định phân tuyến công việc (tiết kiệm hoàn toàn 1 lượt gọi LLM bên dưới).
- **Rút ngắn câu trả lời (Prompt tuning)**:
  - Thêm ràng buộc cụ thể trong system prompt tại [tax_agent/graph.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/tax_agent/graph.py) yêu cầu chuyên gia thuế trả lời cực kỳ ngắn gọn dưới 100 từ, giúp tiết kiệm lượng token tiêu thụ và đẩy nhanh tốc độ phản hồi.

### C. Cải tiến hạ tầng chạy thực tế và Scripts
- **Sử dụng địa chỉ IPv4 tường minh**:
  - Đổi toàn bộ các tham chiếu `localhost` thành `127.0.0.1` trong tệp cấu hình, registry client, và client thử nghiệm. Điều này giúp loại bỏ hoàn toàn hiện tượng trễ hoặc lỗi kết nối do cơ chế phân giải tên miền (DNS lookup) không nhất quán trên Windows.
- **Hỗ trợ đầy đủ môi trường Windows**:
  - Viết tệp kịch bản PowerShell [start_all.ps1](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/start_all.ps1) quản lý chạy song song 5 microservices. Kịch bản này kiểm tra xác thực OpenRouter trước, khởi chạy ẩn các tiến trình Python qua `uv run`, tự động kiểm tra endpoint `/health` hoặc `.well-known/agent.json` để xác định trạng thái sẵn sàng của dịch vụ, lưu log vào thư mục `.stage5-logs/`, và tự động tắt sạch cây tiến trình con một cách an toàn khi nhấn `Ctrl+C`.
- **Cải thiện độ tin cậy của Test Client**:
  - Bổ sung cơ chế trích xuất và in thông báo lỗi chi tiết trong [test_client.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/test_client.py) thay vì chỉ in ra phản hồi thô khi request thất bại, giúp việc debug cấu hình API Key dễ dàng hơn nhiều.
- **Trực quan hóa phân tán**:
  - Tạo tệp HTML demo phân tán [docs/stage_5_a2a_interactive_demo.html](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/docs/stage_5_a2a_interactive_demo.html) tương tác cực kỳ sinh động.

---

## 6. Tổng kết bảng so sánh sự thay đổi

| Tính năng / Yêu cầu | Trước khi sửa đổi | Sau khi sửa đổi | File mã nguồn liên quan |
|---|---|---|---|
| **Cấu hình & Validate LLM** | Không có kiểm tra sớm, `temperature` mặc định, `max_tokens` nhỏ | Validate API Key khi chạy, `temperature=0.3`, `max_tokens=800` | [common/llm.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/llm.py) |
| **Cơ sở kiến thức RAG** | Chỉ chứa các điều luật nước ngoài | Thêm Luật Lao động VN 2019 | [stages/stage_2_rag_tools/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_2_rag_tools/main.py) |
| **Công cụ Thời hiệu** | Chưa có | Công cụ `check_statute_of_limitations` | [stages/stage_2_rag_tools/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_2_rag_tools/main.py), [exercises/exercise_2_tools.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/exercises/exercise_2_tools.py) |
| **Công cụ Án lệ & Debug** | Chưa có | Công cụ `search_case_law` & bật `debug=True` trong ReAct Agent | [stages/stage_3_single_agent/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_3_single_agent/main.py) |
| **Privacy Specialist Agent** | Chưa có | Thêm `privacy_agent`, tự động định tuyến khi phát hiện từ khóa | [stages/stage_4_milti_agent/main.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/stages/stage_4_milti_agent/main.py), [exercises/exercise_4_multiagent.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/exercises/exercise_4_multiagent.py) |
| **HTTP Client & Cache** | Khởi tạo HTTP Client liên tục, không cache card/endpoint | Tái sử dụng HTTP Client, lưu cache card và registry discovery | [common/a2a_client.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/a2a_client.py), [common/registry_client.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/common/registry_client.py) |
| **Cấu trúc Đồ thị Agent** | Sử dụng ReAct Agent cho toàn bộ Agent, chia nhỏ nhiều node | Gộp node `analyze_and_route`, chuyển Tax và Compliance Agent sang đồ thị tĩnh 1 node | [law_agent/graph.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/law_agent/graph.py), [tax_agent/graph.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/tax_agent/graph.py), [compliance_agent/graph.py](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/compliance_agent/graph.py) |
| **Kịch bản Windows** | Chỉ có Shell script (`.sh`) cho Linux/macOS | Thêm PowerShell script (`.ps1`) kiểm tra sẵn sàng và dọn dẹp an toàn | [start_all.ps1](file:///d:/user/Desktop/Github/Batch02-Day9_Multi-Agent_MCP-A2A-Vinh/start_all.ps1) |
