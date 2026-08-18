from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import SecretStr, ValidationError

from core import settings
from tools.elasticsearch_search import (
    DateRange,
    LegalDocumentSearchInput,
    _build_search_body,
    _format_hits,
    _search_legal_documents,
)


def test_date_range_requires_a_bound():
    with pytest.raises(ValidationError):
        DateRange()


def test_build_search_body_uses_search_fields_and_filters():
    search_input = LegalDocumentSearchInput(
        query="đơn phương chấm dứt hợp đồng",
        search_reason="Tìm văn bản theo từ khóa và các filter được chỉ định.",
        ngay_ban_hanh=DateRange(tu="2019-01-01", den="2024-12-31"),
        ngay_co_hieu_luc=DateRange(tu="2021-01-01"),
        tinh_trang_hieu_luc="Còn hiệu lực",
        loai_van_ban=["Luật", "Bộ luật"],
        so_hieu="45/2019/QH14",
        don_vi="Trung ương",
        limit=5,
    )

    body = _build_search_body(search_input)

    multi_match = body["query"]["bool"]["must"][0]["multi_match"]
    assert multi_match["fields"] == ["title^3", "cleaned_toan_van"]
    assert body["size"] == 5
    assert body["_source"] == {"excludes": ["toan_van", "cleaned_toan_van", "html_with_reference"]}
    filters = body["query"]["bool"]["filter"]
    assert len(filters) == 6
    document_type_filter = filters[3]
    assert document_type_filter["bool"]["minimum_should_match"] == 1
    type_options = document_type_filter["bool"]["should"]
    assert [
        option["bool"]["should"][1]["term"]["loai_van_ban.keyword"]["value"]
        for option in type_options
    ] == ["Luật", "Bộ luật"]
    assert body["query"]["bool"]["filter"][-1] == {
        "bool": {
            "should": [
                {
                    "term": {
                        "don_vi": {
                            "value": "Trung ương",
                            "case_insensitive": True,
                        }
                    }
                },
                {
                    "term": {
                        "don_vi.keyword": {
                            "value": "Trung ương",
                            "case_insensitive": True,
                        }
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


def test_legal_document_search_schema_limits_don_vi_to_central_documents():
    schema = LegalDocumentSearchInput.model_json_schema()

    assert "don_vi" in schema["properties"]
    description = schema["properties"]["don_vi"]["description"]
    assert 'don_vi="Trung ương"' in description
    assert "Không dùng trường này để lọc văn bản địa phương" in description


def test_legal_document_search_requires_search_reason():
    schema = LegalDocumentSearchInput.model_json_schema()

    assert "search_reason" in schema["required"]
    assert "Không nêu kết luận chưa được" in schema["properties"]["search_reason"]["description"]


@pytest.mark.parametrize("value", ["Biên bản không có trong danh mục", []])
def test_legal_document_search_rejects_unknown_or_empty_document_types(value):
    with pytest.raises(ValidationError):
        LegalDocumentSearchInput(
            query="luật ban hành",
            search_reason="Kiểm tra danh mục loại văn bản.",
            loai_van_ban=value,
        )


def test_format_hits_returns_document_info_metadata_and_structure():
    payload = {
        "hits": {
            "hits": [
                {
                    "_id": "999999999456223",
                    "_index": "law_documents",
                    "_score": 12.5,
                    "_source": {
                        "title": "Quyết định số 4469/QĐ-BYT",
                        "so_hieu": "4469/QĐ-BYT",
                        "toan_van": "nội dung rất dài",
                        "cleaned_toan_van": "nội dung thuần văn bản",
                        "html_with_reference": "<p>nội dung hiển thị</p>",
                        "cau_truc_van_ban": [{"dieu": 1, "title": "Điều 1"}],
                    },
                    "highlight": {"cleaned_toan_van": ["<mark>nội dung</mark>"]},
                }
            ]
        }
    }

    results = _format_hits(payload)

    assert results == [
        {
            "doc_info": {
                "id": "999999999456223",
                "index": "law_documents",
                "score": 12.5,
            },
            "title": "Quyết định số 4469/QĐ-BYT",
            "metadata": {"so_hieu": "4469/QĐ-BYT"},
            "cau_truc_van_ban": [{"dieu": 1, "title": "Điều 1"}],
            "highlights": {"cleaned_toan_van": ["<mark>nội dung</mark>"]},
        }
    ]


@pytest.mark.asyncio
async def test_search_calls_configured_elasticsearch_endpoint():
    response = Mock()
    response.json.return_value = {"hits": {"hits": []}}
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.post.return_value = response
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = client

    with (
        patch.object(settings, "ELK_ENDPOINT", "https://elk.example.com/"),
        patch.object(settings, "ELK_DOCUMENT_INDEX", "law_documents"),
        patch.object(settings, "ELK_USERNAME", "elastic"),
        patch.object(settings, "ELK_PASSWORD", SecretStr("secret")),
        patch("tools.elasticsearch_search.httpx.AsyncClient", return_value=context_manager),
    ):
        result = await _search_legal_documents(
            LegalDocumentSearchInput(query="thử việc", search_reason="Kiểm thử truy vấn.")
        )

    assert result == []
    client.post.assert_awaited_once()
    assert client.post.await_args.args[0] == "https://elk.example.com/law_documents/_search"


@pytest.mark.asyncio
async def test_search_requires_endpoint_and_index():
    with (
        patch.object(settings, "ELK_ENDPOINT", None),
        patch.object(settings, "ELK_DOCUMENT_INDEX", None),
    ):
        with pytest.raises(RuntimeError, match="ELK_ENDPOINT, ELK_DOCUMENT_INDEX"):
            await _search_legal_documents(
                LegalDocumentSearchInput(query="thử việc", search_reason="Kiểm thử cấu hình.")
            )
