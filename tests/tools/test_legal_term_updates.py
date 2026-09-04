from unittest.mock import MagicMock, Mock, patch

from tools.legal_term_updates import (
    FIND_LATEST_TERMS_QUERY,
    UPDATE_RELATIONSHIPS,
    _find_latest_terms,
    neo4j_term_to_search_result,
)


def test_neo4j_query_follows_all_supported_update_relationships() -> None:
    assert "<-[relationships*1..]-(updated:DIEU_KHOAN)" in FIND_LATEST_TERMS_QUERY
    assert UPDATE_RELATIONSHIPS == ["thay_the", "bai_bo", "sua_doi", "sua_doi_bo_sung", "bo_sung"]


def test_find_latest_terms_uses_parameterized_term_ids() -> None:
    record = {
        "source_term_id": "12083765",
        "term": {"ID": 12090000, "noi_dung": "Điều khoản mới"},
    }
    session = Mock()
    session.run.return_value = [record]
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    with (
        patch("tools.legal_term_updates._get_neo4j_config", return_value=("bolt://x", "u", "p", "neo4j")),
        patch("tools.legal_term_updates._get_neo4j_driver", return_value=driver),
    ):
        assert _find_latest_terms(["12083765"]) == [
            {**record["term"], "_source_term_id": "12083765"}
        ]

    session.run.assert_called_once_with(
        FIND_LATEST_TERMS_QUERY,
        term_ids=["12083765"],
        relation_types=UPDATE_RELATIONSHIPS,
    )


def test_neo4j_term_conversion_preserves_search_result_shape() -> None:
    result = neo4j_term_to_search_result(
        {
            "ID": 12090000,
            "dieu_1": "dieu_1#999999999719107",
            "noi_dung": "Điều khoản mới",
            "so_hieu": "1/2026/NĐ-CP",
        }
    )

    assert result["term_id"] == 12090000
    assert result["doc_id"] == "999999999719107"
    assert result["embed_content"] == "Điều khoản mới"
    assert result["update_source"] == "neo4j"
