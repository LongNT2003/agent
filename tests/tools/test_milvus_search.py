from unittest.mock import Mock, patch

from tools.milvus_search import (
    LawTermSearchInput,
    LawTitleSearchInput,
    _build_filter,
    _format_search_results,
    _search_collection,
)


def test_build_filter_supports_metadata_and_document_id():
    search_input = LawTermSearchInput(
        query="người lao động",
        tinh_trang_hieu_luc="Còn hiệu lực",
        so_hieu='45/2019/"QH14',
        doc_id=123,
    )

    assert _build_filter(search_input) == (
        'tinh_trang_hieu_luc == "Còn hiệu lực" and so_hieu == "45/2019/"QH14" and doc_id == 123'
    )


def test_search_collection_uses_cosine_vector_search_and_filter():
    client = Mock()
    client.search.return_value = [[]]
    search_input = LawTitleSearchInput(query="thuế thu nhập", id_document=456, limit=5)

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


def test_format_term_results_returns_content_metadata_and_score():
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
