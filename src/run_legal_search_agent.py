"""Run the legal search agent once or in an interactive terminal session."""

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy legal-search-agent trong terminal.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Query chạy một lần rồi thoát.")
    mode.add_argument(
        "--session",
        action="store_true",
        help="Nhập nhiều query; gõ exit, quit hoặc q để thoát.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Hiện từng loop, tool call và kết quả retrieval rút gọn.",
    )
    return parser.parse_args()


def load_legal_search_agent() -> Any:
    module_path = Path(__file__).parent / "agents" / "legal_search_agent.py"
    spec = importlib.util.spec_from_file_location("standalone_legal_search_agent", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể load agent từ {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.legal_search_agent


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def format_output(content: Any) -> str:
    text = message_text(content).strip()
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return text


def compact_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def parse_tool_output(content: Any) -> Any:
    text = message_text(content).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def print_tool_results(message: ToolMessage, loop: int) -> None:
    payload = parse_tool_output(message.content)
    tool_name = message.name or "unknown_tool"
    if not isinstance(payload, list):
        print(f"[loop {loop}] {tool_name} result:")
        print(compact_text(payload, limit=1000))
        return

    print(f"[loop {loop}] {tool_name} trả {len(payload)} kết quả:")
    for rank, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            print(f"  {rank}. {compact_text(item)}")
            continue
        doc_info = item.get("doc_info") or {}
        document_id = (
            doc_info.get("id") or item.get("doc_id") or item.get("id_document") or item.get("id")
        )
        score = doc_info.get("score") if doc_info else item.get("score")
        title = (
            item.get("title")
            or item.get("term_title")
            or item.get("article_title")
            or item.get("trich_yeu")
            or item.get("embed_content")
        )
        score_text = f", score={score}" if score is not None else ""
        print(f"  {rank}. id={document_id}{score_text} | {compact_text(title)}")


def print_tool_calls(message: AIMessage, loop: int) -> None:
    for call in message.tool_calls:
        print(f"[loop {loop}] gọi {call['name']}")
        print(json.dumps(call.get("args", {}), ensure_ascii=False, indent=2))


async def run_query_with_debug(agent: Any, query: str, config: dict[str, Any]) -> str:
    loop = 0
    final_message: AIMessage | None = None
    async for update in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        config,
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue
        model_update = update.get("model") or {}
        for message in model_update.get("messages", []):
            if not isinstance(message, AIMessage):
                continue
            if message.tool_calls:
                if loop:
                    print(f"[decision] Chưa đủ bằng chứng, tiếp tục search loop {loop + 1}.")
                loop += 1
                print_tool_calls(message, loop)
            else:
                final_message = message
                print("[decision] Dừng search và trả văn bản đích.")

        tools_update = update.get("tools") or {}
        for message in tools_update.get("messages", []):
            if isinstance(message, ToolMessage):
                print_tool_results(message, loop)

    if final_message is None:
        raise RuntimeError("Agent không trả về final message.")
    return format_output(final_message.content)


async def run_query(agent: Any, query: str, debug: bool = False) -> str:
    config = {"recursion_limit": settings.LEGAL_SEARCH_MAX_LOOPS * 2 + 4}
    if debug:
        return await run_query_with_debug(agent, query, config)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config,
    )
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Agent không trả về message nào.")
    return format_output(messages[-1].content)


async def run_session(agent: Any, debug: bool = False) -> None:
    print("Legal search session. Gõ exit, quit hoặc q để thoát.")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"exit", "quit", "q"}:
            return
        if not query:
            continue
        try:
            print(await run_query(agent, query, debug=debug))
        except Exception as exc:
            print(f"Lỗi: {exc}", file=sys.stderr)


async def main() -> None:
    args = parse_args()
    agent = load_legal_search_agent()
    if args.query is not None:
        print(await run_query(agent, args.query, debug=args.debug))
        return
    await run_session(agent, debug=args.debug)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    asyncio.run(main())
