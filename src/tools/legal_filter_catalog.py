"""Shared legal-search filter catalogs loaded from JSON data files."""

import json
from pathlib import Path


def _load_string_catalog(filename: str, label: str) -> list[str]:
    path = Path(__file__).parents[1] / "data" / filename
    with path.open(encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise RuntimeError(f"Danh mục {label} không hợp lệ: {path}")
    return list(dict.fromkeys(values))


LEGAL_DOCUMENT_TYPES = _load_string_catalog("loai_van_ban.json", "loại văn bản")
LEGAL_DOCUMENT_TYPE_SET = frozenset(LEGAL_DOCUMENT_TYPES)


def validate_legal_document_types(value: str | list[str] | None) -> str | list[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if not values:
        raise ValueError("Danh sách loại văn bản không được rỗng.")
    invalid_values = [item for item in values if item not in LEGAL_DOCUMENT_TYPE_SET]
    if invalid_values:
        invalid_text = ", ".join(invalid_values)
        raise ValueError(f"Loại văn bản không có trong danh mục hiện hành: {invalid_text}")
    return value
