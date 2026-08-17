from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from tools.milvus_search import (
    LawTermInDocumentSearchInput,
    LawTermSearchInput,
    LawTitleSearchInput,
    _build_filter,
    _format_search_results,
    _search_collection,
    search_law_terms,
    search_law_terms_in_document,
)


def test_global_term_search_supports_metadata_but_not_document_id() -> None:
    search_input = LawTermSearchInput(
        query="người lao động",
        search_reason="Tìm điều khoản theo ngữ nghĩa với các filter được chỉ định.",
        tinh_trang_hieu_luc="Còn hiệu lực",
        so_hieu="45/2019/QH14",
    )

    assert _build_filter(search_input) == (
        'tinh_trang_hieu_luc == "Còn hiệu lực" and so_hieu == "45/2019/QH14"'
    )
    assert "doc_id" not in search_law_terms.args

    with pytest.raises(ValidationError):
        LawTermSearchInput(
            query="người lao động",
            search_reason="Kiểm tra schema không nhận doc_id.",
            doc_id=123,
        )


def test_milvus_search_requires_search_reason() -> None:
    schema = LawTermSearchInput.model_json_schema()

    assert "search_reason" in schema["required"]
    assert "Không nêu kết luận chưa được" in schema["properties"]["search_reason"]["description"]


def test_document_term_search_requires_and_filters_document_id() -> None:
    search_input = LawTermInDocumentSearchInput(
        query="người lao động",
        doc_id=123,
        search_reason="Kiểm chứng điều khoản trong candidate đã biết.",
    )

    assert _build_filter(search_input) == "doc_id == 123"
    assert "default" not in search_law_terms_in_document.args["doc_id"]


def test_document_term_search_filters_multiple_document_ids_with_in() -> None:
    search_input = LawTermInDocumentSearchInput(
        query="người lao động",
        doc_id=[123, 456, 789],
        search_reason="Kiểm chứng điều khoản trong nhiều candidate đã biết.",
    )

    assert _build_filter(search_input) == "doc_id in [123, 456, 789]"


@pytest.mark.parametrize("doc_ids", [[], list(range(11))])
def test_document_term_search_rejects_invalid_document_id_list_size(doc_ids: list[int]) -> None:
    with pytest.raises(ValidationError):
        LawTermInDocumentSearchInput(
            query="người lao động",
            doc_id=doc_ids,
            search_reason="Kiểm chứng danh sách candidate.",
        )


def test_search_collection_uses_cosine_vector_search_and_filter() -> None:
    client = Mock()
    client.search.return_value = [[]]
    search_input = LawTitleSearchInput(
        query="thuế thu nhập",
        search_reason="Kiểm thử tìm tiêu đề theo ngữ nghĩa.",
        id_document=456,
        limit=5,
    )

    with (
        patch("tools.milvus_search._load_collection"),
        patch("tools.milvus_search._get_milvus_client", return_value=client),
        patch("tools.milvus_search._embed_query", return_value=[0.1, 0.2]),
    ):
        result = _search_collection(search_input, "law_title_CMC", "title")

    assert result == []
    client.search.assert_called_once()
    kwargs = client.search.call_args.kwargs
    assert kwargs["collection_name"] == "law_title_CMC"
    assert kwargs["data"] == [[0.1, 0.2]]
    assert kwargs["anns_field"] == "dense_vector"
    assert kwargs["search_params"]["metric_type"] == "COSINE"
    assert kwargs["filter"] == "id_document == 456"
    assert kwargs["limit"] == 5


def test_format_term_results_returns_content_metadata_and_score() -> None:
    values = {
        "term_id": "term-1",
        "doc_id": 99,
        "position": "Điều 3",
        "term_title": "Người lao động",
        "article_title": "Giải thích từ ngữ",
        "embed_content": "Người lao động là...",
        "part_index": 0,
        "so_hieu": "45/2019/QH14",
        "tinh_trang_hieu_luc": "Còn hiệu lực",
        "co_quan_ban_hanh": "Quốc hội",
        "loai_van_ban": "Bộ luật",
    }
    hit = {"id": 7, "distance": 0.91, "entity": values}

    result = _format_search_results([[hit]], "law_terms_CMC", "term")

    assert result[0]["id"] == 7
    assert result[0]["score"] == 0.91
    assert result[0]["term_title"] == "Người lao động"
    assert result[0]["metadata"]["so_hieu"] == "45/2019/QH14"
