'''Elasticsearch BM25 search tool for Vietnamese legal documents.'''

from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, model_validator

from core import settings


class DateRange(BaseModel):
    '''Inclusive date range accepted by an Elasticsearch date field.'''

    tu: str | None = Field(default=None, description='Ngày bắt đầu, định dạng YYYY-MM-DD.')
    den: str | None = Field(default=None, description='Ngày kết thúc, định dạng YYYY-MM-DD.')

    @model_validator(mode='after')
    def validate_bounds(self) -> 'DateRange':
        if not self.tu and not self.den:
            raise ValueError('Khoảng ngày phải có ít nhất một trong hai giá trị tu hoặc den.')
        return self


class LegalDocumentSearchInput(BaseModel):
    '''Input schema for legal document BM25 search.'''

    query: str = Field(
        min_length=1,
        description='Từ khóa cần tìm trong title và toàn văn thuần văn bản.',
    )
    search_reason: str = Field(
        min_length=1,
        description=(
            'Lý do ngắn trước khi search: intent cần tìm, vì sao chọn Elasticsearch, '
            'và vì sao dùng hoặc không dùng từng filter. Không nêu kết luận chưa được '
            'kết quả tool xác nhận.'
        ),
    )
    ngay_ban_hanh: DateRange | None = Field(
        default=None,
        description='Lọc theo khoảng ngày ban hành.',
    )
    ngay_co_hieu_luc: DateRange | None = Field(
        default=None,
        description='Lọc theo khoảng ngày có hiệu lực.',
    )
    tinh_trang_hieu_luc: str | None = Field(
        default=None,
        description='Lọc chính xác tình trạng hiệu lực, ví dụ: Còn hiệu lực.',
    )
    so_hieu: str | None = Field(
        default=None,
        description='Lọc chính xác số hiệu văn bản, ví dụ: 45/2019/QH14.',
    )
    don_vi: str | None = Field(
        default=None,
        description=(
            'Chỉ dùng don_vi="Trung ương" khi cần giới hạn ở văn bản cấp trung ương. '
            'Không dùng trường này để lọc văn bản địa phương vì dữ liệu địa phương '
            'chưa được chuẩn hóa và có nhiều cách biểu diễn.'
        ),
    )
    limit: int = Field(default=10, ge=1, le=50, description='Số văn bản tối đa trả về.')


def _date_range_filter(field: str, value: DateRange) -> dict[str, Any]:
    bounds: dict[str, str] = {}
    if value.tu:
        bounds['gte'] = value.tu
    if value.den:
        bounds['lte'] = value.den
    return {'range': {field: bounds}}


def _exact_filter(field: str, value: str) -> dict[str, Any]:
    '''Support indices where an exact field is either keyword or text+keyword.'''
    return {
        'bool': {
            'should': [
                {'term': {field: {'value': value, 'case_insensitive': True}}},
                {
                    'term': {
                        f'{field}.keyword': {'value': value, 'case_insensitive': True}
                    }
                },
            ],
            'minimum_should_match': 1,
        }
    }


def _build_search_body(search_input: LegalDocumentSearchInput) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []
    if search_input.ngay_ban_hanh:
        filters.append(_date_range_filter('ngay_ban_hanh', search_input.ngay_ban_hanh))
    if search_input.ngay_co_hieu_luc:
        filters.append(
            _date_range_filter('ngay_co_hieu_luc', search_input.ngay_co_hieu_luc)
        )
    if search_input.tinh_trang_hieu_luc:
        filters.append(
            _exact_filter('tinh_trang_hieu_luc', search_input.tinh_trang_hieu_luc)
        )
    if search_input.so_hieu:
        filters.append(_exact_filter('so_hieu', search_input.so_hieu))
    if search_input.don_vi:
        filters.append(_exact_filter('don_vi', search_input.don_vi))

    return {
        'size': search_input.limit,
        'track_total_hits': False,
        '_source': {
            'excludes': ['toan_van', 'cleaned_toan_van', 'html_with_reference']
        },
        'query': {
            'bool': {
                'must': [
                    {
                        'multi_match': {
                            'query': search_input.query,
                            'fields': ['title^3', 'cleaned_toan_van'],
                            'type': 'best_fields',
                            'minimum_should_match': '70%',
                        }
                    }
                ],
                'filter': filters,
            }
        },
        'highlight': {
            'fields': {
                'title': {'number_of_fragments': 0},
                'cleaned_toan_van': {
                    'fragment_size': 300,
                    'number_of_fragments': 3,
                },
            },
            'pre_tags': ['<mark>'],
            'post_tags': ['</mark>'],
        },
    }


def _format_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for hit in payload.get('hits', {}).get('hits', []):
        source = dict(hit.get('_source') or {})
        title = source.pop('title', None)
        structure = source.pop('cau_truc_van_ban', [])
        source.pop('toan_van', None)
        source.pop('cleaned_toan_van', None)
        source.pop('html_with_reference', None)

        results.append(
            {
                'doc_info': {
                    'id': hit.get('_id'),
                    'index': hit.get('_index'),
                    'score': hit.get('_score'),
                },
                'title': title,
                'metadata': source,
                'cau_truc_van_ban': structure,
                'highlights': hit.get('highlight', {}),
            }
        )
    return results


def _get_elk_config() -> tuple[str, str, httpx.BasicAuth | None]:
    missing = [
        name
        for name, value in {
            'ELK_ENDPOINT': settings.ELK_ENDPOINT,
            'ELK_DOCUMENT_INDEX': settings.ELK_DOCUMENT_INDEX,
        }.items()
        if not value
    ]
    if missing:
        separator = ', '
        raise RuntimeError(f'Thiếu cấu hình Elasticsearch: {separator.join(missing)}')

    if bool(settings.ELK_USERNAME) != bool(settings.ELK_PASSWORD):
        raise RuntimeError('ELK_USERNAME và ELK_PASSWORD phải được cấu hình cùng nhau.')

    auth = None
    if settings.ELK_USERNAME and settings.ELK_PASSWORD:
        auth = httpx.BasicAuth(
            username=settings.ELK_USERNAME,
            password=settings.ELK_PASSWORD.get_secret_value(),
        )

    endpoint = settings.ELK_ENDPOINT
    index = settings.ELK_DOCUMENT_INDEX
    if endpoint is None or index is None:
        raise RuntimeError('Cấu hình Elasticsearch không hợp lệ.')
    return endpoint.rstrip('/'), index, auth


async def _search_legal_documents(
    search_input: LegalDocumentSearchInput,
) -> list[dict[str, Any]]:
    endpoint, index, auth = _get_elk_config()
    index_path = quote(index, safe=',*-_')
    try:
        async with httpx.AsyncClient(auth=auth, timeout=15.0) as client:
            response = await client.post(
                f'{endpoint}/{index_path}/_search',
                json=_build_search_body(search_input),
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f'Không thể tìm kiếm văn bản trên Elasticsearch: {exc}'
        ) from exc

    return _format_hits(response.json())


async def search_legal_documents_func(
    query: str,
    search_reason: str,
    ngay_ban_hanh: DateRange | None = None,
    ngay_co_hieu_luc: DateRange | None = None,
    tinh_trang_hieu_luc: str | None = None,
    so_hieu: str | None = None,
    don_vi: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    '''Tìm văn bản pháp luật bằng BM25 trên tiêu đề và toàn văn.

    Có thể lọc theo ngày ban hành, ngày có hiệu lực, tình trạng hiệu lực, số hiệu và
    đơn vị ban hành. Chỉ dùng don_vi="Trung ương" để giới hạn ở văn bản cấp trung
    ương; không dùng don_vi để lọc văn bản địa phương.
    Kết quả gồm thông tin document, metadata, cấu trúc văn bản và các đoạn khớp.
    '''
    search_input = LegalDocumentSearchInput(
        query=query,
        search_reason=search_reason,
        ngay_ban_hanh=ngay_ban_hanh,
        ngay_co_hieu_luc=ngay_co_hieu_luc,
        tinh_trang_hieu_luc=tinh_trang_hieu_luc,
        so_hieu=so_hieu,
        don_vi=don_vi,
        limit=limit,
    )
    return await _search_legal_documents(search_input)


search_legal_documents: BaseTool = tool(
    search_legal_documents_func,
    args_schema=LegalDocumentSearchInput,
)
search_legal_documents.name = 'search_legal_documents'
