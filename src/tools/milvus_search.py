"""Milvus semantic search tools for Vietnamese legal terms and document titles."""

import asyncio
from functools import lru_cache
from threading import Lock
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from core import settings

_embedding_lock = Lock()


class MilvusLegalSearchInput(BaseModel):
    """Common input for legal semantic search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Nội dung pháp luật cần tìm theo ngữ nghĩa.")
    search_reason: str = Field(
        min_length=1,
        description=(
            "Lý do ngắn trước khi search: intent cần tìm, vì sao chọn Milvus, "
            "và vì sao dùng hoặc không dùng từng filter. Không nêu kết luận chưa được "
            "kết quả tool xác nhận."
        ),
    )
    tinh_trang_hieu_luc: str | list[str] | None = Field(
        default=None,
        description=(
            "Lọc chính xác theo một hoặc nhiều tình trạng hiệu lực. Nhiều giá trị "
            "được kết hợp theo OR. Với câu hỏi tình huống hoặc quy tắc hiện hành, "
            'phải truyền đủ ["Còn hiệu lực", "Chưa có hiệu lực", '
            '"Hết hiệu lực một phần"], không rút gọn còn một giá trị.'
        ),
    )
    co_quan_ban_hanh: str | None = Field(
        default=None,
        description="Lọc chính xác cơ quan ban hành.",
    )
    loai_van_ban: str | None = Field(
        default=None,
        description="Lọc chính xác loại văn bản.",
    )
    so_hieu: str | None = Field(default=None, description="Lọc chính xác số hiệu văn bản.")
    limit: int = Field(default=10, ge=1, le=50, description="Số kết quả tối đa trả về.")


class LawTermSearchInput(MilvusLegalSearchInput):
    """Input for searching legal terms."""


class LawTermInDocumentSearchInput(MilvusLegalSearchInput):
    """Input for searching legal terms inside a known candidate document."""

    doc_id: int = Field(description="ID văn bản ứng viên đã tìm thấy ở bước trước.")


class LawTitleSearchInput(MilvusLegalSearchInput):
    """Input for searching legal document titles and summaries."""

    id_document: int | None = Field(default=None, description="Lọc theo ID văn bản.")


def _get_milvus_config() -> tuple[str, int, str | None, str | None]:
    missing = [
        name
        for name, value in {
            "MILVUS_HOST": settings.MILVUS_HOST,
            "MILVUS_PORT": settings.MILVUS_PORT,
        }.items()
        if value is None
    ]
    if missing:
        raise RuntimeError(f"Thiếu cấu hình Milvus: {', '.join(missing)}")

    if bool(settings.MILVUS_USER) != bool(settings.MILVUS_PASSWORD):
        raise RuntimeError("MILVUS_USER và MILVUS_PASSWORD phải được cấu hình cùng nhau.")

    if settings.MILVUS_HOST is None or settings.MILVUS_PORT is None:
        raise RuntimeError("Cấu hình Milvus không hợp lệ.")
    password = settings.MILVUS_PASSWORD.get_secret_value() if settings.MILVUS_PASSWORD else None
    return settings.MILVUS_HOST, settings.MILVUS_PORT, settings.MILVUS_USER, password


@lru_cache(maxsize=1)
def _get_milvus_client() -> Any:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("Chưa cài pymilvus. Hãy chạy `uv sync`.") from exc

    host, port, user, password = _get_milvus_config()
    connect_args: dict[str, Any] = {"uri": f"http://{host}:{port}", "timeout": 60}
    if user and password:
        connect_args.update(user=user, password=password)
    return MilvusClient(**connect_args)


@lru_cache(maxsize=2)
def _load_collection(collection_name: str) -> str:
    client = _get_milvus_client()
    if not client.has_collection(collection_name=collection_name):
        raise RuntimeError(f"Collection Milvus không tồn tại: {collection_name}")
    client.load_collection(collection_name=collection_name)
    return collection_name


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Chưa cài sentence-transformers. Hãy chạy `uv sync`.") from exc

    return SentenceTransformer(
        model_name_or_path=settings.MILVUS_EMBEDDING_MODEL,
        device=settings.MILVUS_EMBEDDING_DEVICE,
    )


def _embed_query(query: str) -> list[float]:
    with _embedding_lock:
        vector = _get_embedding_model().encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
    return vector.astype("float32").tolist()


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '"')


def _build_filter(search_input: MilvusLegalSearchInput) -> str | None:
    expressions = []
    for field in ("tinh_trang_hieu_luc", "co_quan_ban_hanh", "loai_van_ban", "so_hieu"):
        value = getattr(search_input, field)
        if value:
            if isinstance(value, list):
                encoded_values = ", ".join(f'"{_escape_filter_value(item)}"' for item in value)
                expressions.append(f"{field} in [{encoded_values}]")
            else:
                expressions.append(f'{field} == "{_escape_filter_value(value)}"')

    if isinstance(search_input, LawTermInDocumentSearchInput):
        expressions.append(f"doc_id == {search_input.doc_id}")
    if isinstance(search_input, LawTitleSearchInput) and search_input.id_document is not None:
        expressions.append(f"id_document == {search_input.id_document}")
    return " and ".join(expressions) or None


def _format_search_results(
    raw_results: Any,
    collection_name: str,
    result_type: Literal["term", "title"],
) -> list[dict[str, Any]]:
    results = []
    for hit in raw_results[0] if raw_results else []:
        entity = hit["entity"]
        if result_type == "term":
            content = {
                "term_id": entity.get("term_id"),
                "doc_id": entity.get("doc_id"),
                "position": entity.get("position"),
                "term_title": entity.get("term_title"),
                "article_title": entity.get("article_title"),
                "embed_content": entity.get("embed_content"),
                "part_index": entity.get("part_index"),
            }
        else:
            content = {
                "id_document": entity.get("id_document"),
                "trich_yeu": entity.get("trich_yeu"),
                "embed_content": entity.get("embed_content"),
            }
        results.append(
            {
                "id": hit["id"],
                "score": hit["distance"],
                "collection": collection_name,
                **content,
                "metadata": {
                    "so_hieu": entity.get("so_hieu"),
                    "tinh_trang_hieu_luc": entity.get("tinh_trang_hieu_luc"),
                    "co_quan_ban_hanh": entity.get("co_quan_ban_hanh"),
                    "loai_van_ban": entity.get("loai_van_ban"),
                },
            }
        )
    return results


def _search_collection(
    search_input: MilvusLegalSearchInput,
    collection_name: str,
    result_type: Literal["term", "title"],
) -> list[dict[str, Any]]:
    output_fields = [
        "so_hieu",
        "tinh_trang_hieu_luc",
        "co_quan_ban_hanh",
        "loai_van_ban",
    ]
    if result_type == "term":
        output_fields.extend(
            [
                "term_id",
                "doc_id",
                "position",
                "term_title",
                "article_title",
                "embed_content",
                "part_index",
            ]
        )
    else:
        output_fields.extend(["id_document", "trich_yeu", "embed_content"])

    try:
        _load_collection(collection_name)
        raw_results = _get_milvus_client().search(
            collection_name=collection_name,
            data=[_embed_query(search_input.query)],
            anns_field="dense_vector",
            search_params={
                "metric_type": "COSINE",
                "params": {"nprobe": settings.MILVUS_SEARCH_NPROBE},
            },
            limit=search_input.limit,
            filter=_build_filter(search_input) or "",
            output_fields=output_fields,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Không thể tìm kiếm trên Milvus: {exc}") from exc
    return _format_search_results(raw_results, collection_name, result_type)


async def search_law_terms_func(
    query: str,
    search_reason: str,
    tinh_trang_hieu_luc: str | list[str] | None = None,
    co_quan_ban_hanh: str | None = None,
    loai_van_ban: str | None = None,
    so_hieu: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Tìm định nghĩa thuật ngữ pháp luật theo ngữ nghĩa trong law_terms_CMC."""
    search_input = LawTermSearchInput(
        query=query,
        search_reason=search_reason,
        tinh_trang_hieu_luc=tinh_trang_hieu_luc,
        co_quan_ban_hanh=co_quan_ban_hanh,
        loai_van_ban=loai_van_ban,
        so_hieu=so_hieu,
        limit=limit,
    )
    return await asyncio.to_thread(
        _search_collection,
        search_input,
        settings.MILVUS_COLLECTION_TERM_CMC,
        "term",
    )


async def search_law_terms_in_document_func(
    query: str,
    doc_id: int,
    search_reason: str,
    tinh_trang_hieu_luc: str | list[str] | None = None,
    co_quan_ban_hanh: str | None = None,
    loai_van_ban: str | None = None,
    so_hieu: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Tìm điều, khoản trong một văn bản ứng viên đã biết ID."""
    search_input = LawTermInDocumentSearchInput(
        query=query,
        doc_id=doc_id,
        search_reason=search_reason,
        tinh_trang_hieu_luc=tinh_trang_hieu_luc,
        co_quan_ban_hanh=co_quan_ban_hanh,
        loai_van_ban=loai_van_ban,
        so_hieu=so_hieu,
        limit=limit,
    )
    return await asyncio.to_thread(
        _search_collection,
        search_input,
        settings.MILVUS_COLLECTION_TERM_CMC,
        "term",
    )


async def search_law_titles_func(
    query: str,
    search_reason: str,
    tinh_trang_hieu_luc: str | list[str] | None = None,
    co_quan_ban_hanh: str | None = None,
    loai_van_ban: str | None = None,
    so_hieu: str | None = None,
    id_document: int | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Tìm văn bản pháp luật theo ngữ nghĩa tiêu đề/trích yếu trong law_title_CMC."""
    search_input = LawTitleSearchInput(
        query=query,
        search_reason=search_reason,
        tinh_trang_hieu_luc=tinh_trang_hieu_luc,
        co_quan_ban_hanh=co_quan_ban_hanh,
        loai_van_ban=loai_van_ban,
        so_hieu=so_hieu,
        id_document=id_document,
        limit=limit,
    )
    return await asyncio.to_thread(
        _search_collection,
        search_input,
        settings.MILVUS_COLLECTION_TRICH_YEU_CMC,
        "title",
    )


search_law_terms: BaseTool = tool(search_law_terms_func, args_schema=LawTermSearchInput)
search_law_terms.name = "search_law_terms"

search_law_terms_in_document: BaseTool = tool(
    search_law_terms_in_document_func,
    args_schema=LawTermInDocumentSearchInput,
)
search_law_terms_in_document.name = "search_law_terms_in_document"

search_law_titles: BaseTool = tool(search_law_titles_func, args_schema=LawTitleSearchInput)
search_law_titles.name = "search_law_titles"
