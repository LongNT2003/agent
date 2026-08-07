'''Run Elasticsearch or Milvus legal search tools with an inline query.'''

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Literal

# Tool Elasticsearch không cần LLM, nhưng Settings chung của repo yêu cầu một model.
os.environ.setdefault('USE_FAKE_MODEL', 'true')

from tools import (  # noqa: E402, I001
    search_law_terms,
    search_law_titles,
    search_legal_documents,
)


# Sửa query và filter trực tiếp tại đây.
# QUERY = 'đơn phương chấm dứt hợp đồng lao động'
QUERY = 'đi ô tô vượt đèn đỏ bị phạt nhiu tiền'
NGAY_BAN_HANH: dict[str, str] | None = None
NGAY_CO_HIEU_LUC: dict[str, str] | None = None
TINH_TRANG_HIEU_LUC: str | None = 'Còn hiệu lực'
SO_HIEU: str | None = None
LIMIT = 10
SearchType = Literal['elasticsearch', 'term', 'title']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Chạy search văn bản pháp luật trên Elasticsearch hoặc Milvus.'
    )
    parser.add_argument(
        '--search-type',
        choices=('elasticsearch', 'term', 'title'),
        default='elasticsearch',
        help=(
            'elasticsearch: BM25 toàn văn; term: Milvus thuật ngữ; '
            'title: Milvus tiêu đề/trích yếu.'
        ),
    )
    return parser.parse_args()


async def run_search(search_type: SearchType) -> list[dict[str, Any]]:
    common_input = {
        'query': QUERY,
        'tinh_trang_hieu_luc': TINH_TRANG_HIEU_LUC,
        'so_hieu': SO_HIEU,
        'limit': LIMIT,
    }

    if search_type == 'term':
        return await search_law_terms.ainvoke(common_input)
    if search_type == 'title':
        return await search_law_titles.ainvoke(common_input)
    return await search_legal_documents.ainvoke(
        {
            **common_input,
            'ngay_ban_hanh': NGAY_BAN_HANH,
            'ngay_co_hieu_luc': NGAY_CO_HIEU_LUC,
        }
    )


def format_result(
    result: list[dict[str, Any]], search_type: SearchType
) -> list[dict[str, Any]]:
    if search_type == 'term':
        return [
            {
                'ID': item.get('doc_id'),
                'title': item.get('term_title') or item.get('article_title'),
            }
            for item in result
        ]
    if search_type == 'title':
        return [
            {
                'ID': item.get('id_document'),
                'title': item.get('trich_yeu') or item.get('embed_content'),
            }
            for item in result
        ]
    return [
        {
            'ID': item.get('doc_info', {}).get('id'),
            'title': item.get('title'),
        }
        for item in result
    ]


async def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    args = parse_args()
    result = await run_search(args.search_type)
    display_result = format_result(result, args.search_type)
    print(json.dumps(display_result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
