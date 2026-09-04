"""Resolve newer legal terms through amendment relationships in Neo4j."""

import asyncio
from functools import lru_cache
from typing import Any

from core import settings

UPDATE_RELATIONSHIPS = ["thay_the", "bai_bo", "sua_doi", "sua_doi_bo_sung", "bo_sung"]

# Amendment edges point from the newer term to the term that it changes.  Following
# the incoming edges from a search result therefore reaches every newer version.
FIND_LATEST_TERMS_QUERY = """
MATCH (current:DIEU_KHOAN)
WHERE toString(current.ID) IN $term_ids
MATCH path=(current)<-[relationships*1..]-(updated:DIEU_KHOAN)
WHERE ALL(relationship IN relationships WHERE type(relationship) IN $relation_types)
  AND NOT EXISTS {
    MATCH (updated)<-[next_relationship]-(:DIEU_KHOAN)
    WHERE type(next_relationship) IN $relation_types
  }
RETURN DISTINCT toString(current.ID) AS source_term_id, properties(updated) AS term
"""


def _get_neo4j_config() -> tuple[str, str, str, str]:
    missing = [
        name
        for name, value in {
            "NEO4J_HOST": settings.NEO4J_HOST,
            "NEO4J_PORT": settings.NEO4J_PORT,
            "NEO4J_USERNAME": settings.NEO4J_USERNAME,
            "NEO4J_PASSWORD": settings.NEO4J_PASSWORD,
        }.items()
        if value is None
    ]
    if missing:
        raise RuntimeError(f"Thiếu cấu hình Neo4j: {', '.join(missing)}")

    password = settings.NEO4J_PASSWORD.get_secret_value() if settings.NEO4J_PASSWORD else None
    if (
        settings.NEO4J_HOST is None
        or settings.NEO4J_PORT is None
        or settings.NEO4J_USERNAME is None
        or password is None
    ):
        raise RuntimeError("Cấu hình Neo4j không hợp lệ.")
    return (
        f"bolt://{settings.NEO4J_HOST}:{settings.NEO4J_PORT}",
        settings.NEO4J_USERNAME,
        password,
        settings.NEO4J_DATABASE,
    )


@lru_cache(maxsize=1)
def _get_neo4j_driver() -> Any:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Chưa cài neo4j. Hãy chạy `uv sync`.") from exc

    uri, username, password, _ = _get_neo4j_config()
    return GraphDatabase.driver(uri, auth=(username, password))


def _find_latest_terms(term_ids: list[str]) -> list[dict[str, Any]]:
    if not term_ids:
        return []
    _, _, _, database = _get_neo4j_config()
    try:
        with _get_neo4j_driver().session(database=database) as session:
            records = session.run(
                FIND_LATEST_TERMS_QUERY,
                term_ids=term_ids,
                relation_types=UPDATE_RELATIONSHIPS,
            )
            return [
                {**dict(record["term"]), "_source_term_id": record["source_term_id"]}
                for record in records
            ]
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Không thể tìm điều khoản cập nhật trên Neo4j: {exc}") from exc


async def find_latest_terms(term_ids: list[str]) -> list[dict[str, Any]]:
    """Find terminal terms that replace, repeal, amend, or supplement the given terms."""
    return await asyncio.to_thread(_find_latest_terms, term_ids)


def neo4j_term_to_search_result(term: dict[str, Any]) -> dict[str, Any]:
    """Convert a DIEU_KHOAN property map to the result shape used by Milvus tools."""
    term_id = term.get("ID")
    document_id = term.get("id_van_ban")
    if document_id is None and isinstance(term.get("dieu_1"), str):
        document_id = term["dieu_1"].rsplit("#", 1)[-1]
    content = term.get("noi_dung") or term.get("embed_content")
    return {
        "id": term_id,
        "term_id": term_id,
        "doc_id": document_id,
        "position": term.get("vi_tri") or term.get("cap_do"),
        "term_title": term.get("name") or term.get("tieu_de"),
        "article_title": term.get("dieu") or term.get("dieu_1"),
        "embed_content": content,
        "part_index": term.get("part_index"),
        "metadata": {
            "so_hieu": term.get("so_hieu"),
            "tinh_trang_hieu_luc": term.get("tinh_trang_hieu_luc"),
            "co_quan_ban_hanh": term.get("co_quan_ban_hanh"),
            "loai_van_ban": term.get("loai_van_ban"),
        },
        "update_source": "neo4j",
    }
