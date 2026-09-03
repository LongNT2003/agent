# Legal Search Agent

Agent truy hồi văn bản pháp luật Việt Nam bằng tìm kiếm kết hợp **Elasticsearch (BM25)** và **Milvus (semantic search)**. Agent sử dụng Gemini để chuẩn hóa câu hỏi, tìm văn bản ứng viên, kiểm chứng điều khoản trong từng văn bản và trả về danh sách văn bản phù hợp dưới dạng JSON.

> [!IMPORTANT]
> Dự án này là công cụ **tìm kiếm văn bản**, không phải hệ thống tư vấn pháp lý. Agent không kết luận mức phạt, quyền, nghĩa vụ hay cách xử lý một vụ việc. Kết quả cần được đối chiếu với nguồn văn bản chính thức và người có chuyên môn trước khi sử dụng.

## Agent làm gì?

Ví dụ với câu hỏi đời thường như “ô tô vượt đèn đỏ bị phạt theo văn bản nào?”, agent sẽ:

1. Chuẩn hóa truy vấn sang cách diễn đạt pháp lý phù hợp.
2. Tìm văn bản ứng viên bằng BM25 trên Elasticsearch hoặc tìm kiếm ngữ nghĩa trên Milvus.
3. Dùng ID ứng viên đã tìm thấy để kiểm chứng điều, khoản bên trong đúng văn bản đó.
4. Loại các kết quả chỉ khớp từ khóa chung hoặc không đúng đối tượng/hành vi.
5. Trả về các văn bản đủ bằng chứng, hoặc danh sách rỗng nếu chưa tìm được kết quả đáng tin cậy.

Đầu ra có dạng:

```json
{
  "documents": [
    {
      "id": "document-id",
      "title": "Tên văn bản",
      "so_hieu": "Số hiệu văn bản",
      "tinh_trang_hieu_luc": "Còn hiệu lực"
    }
  ]
}
```

## Kiến trúc

```text
Người dùng
    │
    ▼
Gemini + LangGraph
    │
    ├── Elasticsearch ── BM25 trên tiêu đề và toàn văn
    │
    └── Milvus ───────── semantic search trên điều, khoản
                              │
                              └── kiểm chứng theo doc_id
    │
    ▼
Danh sách văn bản JSON
```

Luồng agent có giới hạn số vòng tìm kiếm để tránh lặp vô hạn. Mặc định là 3 vòng và có thể cấu hình từ 1 đến 10 vòng qua `LEGAL_SEARCH_MAX_LOOPS`.

## Công nghệ chính

- Python 3.12–3.14
- LangGraph và LangChain
- Google Gemini
- Elasticsearch
- Milvus và Sentence Transformers
- FastAPI cho HTTP API
- Streamlit cho giao diện chat

## Yêu cầu

- Python 3.12 trở lên
- [uv](https://docs.astral.sh/uv/) để cài dependency
- Một Elasticsearch index chứa dữ liệu văn bản pháp luật
- Một Milvus collection chứa embedding của điều, khoản
- Gemini API key

Elasticsearch và Milvus **không được khởi tạo dữ liệu tự động** bởi repository này. Bạn cần chuẩn bị sẵn index/collection tương thích và thông tin kết nối.

## Cài đặt

```bash
git clone https://github.com/LongNT2003/agent.git
cd agent
uv sync --frozen
```

Tạo file `.env` từ file mẫu:

```bash
cp .env.example .env
```

Trên PowerShell:

```powershell
Copy-Item .env.example .env
```

Thiết lập tối thiểu các biến sau:

```dotenv
# Gemini
GEMINI_API=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

# Elasticsearch
ELK_ENDPOINT=http://localhost:9200
ELK_DOCUMENT_INDEX=your_legal_document_index
ELK_USERNAME=
ELK_PASSWORD=

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_COLLECTION_TERM_CMC=law_terms_CMC

# Embedding và agent
MILVUS_EMBEDDING_MODEL=CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v2
MILVUS_EMBEDDING_DEVICE=cpu
MILVUS_SEARCH_NPROBE=16
LEGAL_SEARCH_MAX_LOOPS=3
```

`ELK_USERNAME` và `ELK_PASSWORD` phải cùng được khai báo hoặc cùng để trống. Quy tắc tương tự áp dụng cho `MILVUS_USER` và `MILVUS_PASSWORD`.

Lần chạy semantic search đầu tiên có thể mất thêm thời gian để tải embedding model.

## Chạy trong terminal

Chạy một truy vấn rồi thoát:

```bash
uv run python src/run_legal_search_agent.py --query "Ô tô vượt đèn đỏ bị xử phạt theo văn bản nào?"
```

Hiện từng vòng tìm kiếm, tool call và kết quả retrieval rút gọn:

```bash
uv run python src/run_legal_search_agent.py --query "Ô tô vượt đèn đỏ bị xử phạt theo văn bản nào?" --debug
```

Chạy phiên tương tác:

```bash
uv run python src/run_legal_search_agent.py --session
```

Gõ `exit`, `quit` hoặc `q` để thoát.

## Chạy API và giao diện

Khởi động FastAPI service:

```bash
uv run python src/run_service.py
```

API mặc định chạy tại `http://localhost:8080`. Kiểm tra danh sách agent và model:

```bash
curl http://localhost:8080/info
```

Gọi legal search agent:

```bash
curl -X POST http://localhost:8080/legal-search-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"message":"Văn bản nào quy định xử phạt ô tô vượt đèn đỏ?"}'
```

Endpoint streaming tương ứng là:

```text
POST /legal-search-agent/stream
```

Để chạy giao diện Streamlit, mở terminal khác:

```bash
uv run streamlit run src/streamlit_app.py
```

Sau đó truy cập `http://localhost:8501` và chọn `legal-search-agent`.

## Chạy bằng Docker

Sau khi đã cấu hình `.env`:

```bash
docker compose watch
```

Các service mặc định:

- FastAPI: `http://localhost:8080`
- Streamlit: `http://localhost:8501`
- PostgreSQL: `localhost:5432` (dịch vụ lưu trữ tùy chọn của toolkit)

Elasticsearch và Milvus không nằm trong `compose.yaml`; container agent phải truy cập được các dịch vụ bạn cấu hình trong `.env`.

## Kiểm thử

Chạy toàn bộ test suite:

```bash
uv run pytest
```

Chỉ chạy test liên quan đến legal search:

```bash
uv run pytest tests/agents/test_legal_search_agent.py tests/tools/test_elasticsearch_search.py tests/tools/test_milvus_search.py
```

## Cấu trúc liên quan

```text
src/
├── agents/
│   └── legal_search_agent.py       # LangGraph workflow và luật điều phối tìm kiếm
├── tools/
│   ├── elasticsearch_search.py     # Tìm văn bản bằng BM25
│   └── milvus_search.py            # Tìm kiếm ngữ nghĩa và kiểm chứng theo doc_id
├── run_legal_search_agent.py       # CLI chạy riêng legal search agent
├── run_service.py                  # FastAPI service
└── streamlit_app.py                # Giao diện chat

tests/
├── agents/test_legal_search_agent.py
└── tools/
    ├── test_elasticsearch_search.py
    └── test_milvus_search.py
```

## Giới hạn hiện tại

- Chất lượng kết quả phụ thuộc trực tiếp vào dữ liệu và metadata trong Elasticsearch/Milvus.
- Agent chỉ trả văn bản đích; không trích dẫn hay giải thích đầy đủ nội dung pháp luật.
- Trạng thái hiệu lực lấy từ dữ liệu đã index và có thể không phản ánh thay đổi pháp luật mới nhất nếu dữ liệu chưa được cập nhật.
- Embedding model mặc định chạy trên CPU; có thể đổi `MILVUS_EMBEDDING_DEVICE` nếu môi trường hỗ trợ accelerator tương thích.

## License

Dự án được phát hành theo giấy phép [MIT](LICENSE).
