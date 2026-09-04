"""Run the legal search agent once or in an interactive terminal session."""

import argparse
import asyncio
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core import settings

logger = logging.getLogger("legal_search_agent_runner")


def configure_file_logging() -> Path:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"legal_search_agent_{timestamp}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    logging.captureWarnings(True)
    logger.debug("Bắt đầu legal search runner. log_file=%s", log_path)
    return log_path


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


def emit_debug(text: str, show_debug: bool) -> None:
    logger.debug("%s", text)
    if show_debug:
        print(text)


def print_tool_results(message: ToolMessage, loop: int, show_debug: bool) -> None:
    payload = parse_tool_output(message.content)
    tool_name = message.name or "unknown_tool"
    if not isinstance(payload, list):
        emit_debug(
            f"[loop {loop}] {tool_name} result:\n{compact_text(payload, limit=1000)}",
            show_debug,
        )
        return

    lines = [f"[loop {loop}] {tool_name} trả {len(payload)} kết quả:"]
    for rank, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            lines.append(f"  {rank}. {compact_text(item)}")
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
        lines.append(f"  {rank}. id={document_id}{score_text} | {compact_text(title)}")
    emit_debug("\n".join(lines), show_debug)


def print_tool_calls(message: AIMessage, loop: int, show_debug: bool) -> None:
    for call in message.tool_calls:
        emit_debug(
            f"[loop {loop}] gọi {call['name']}\n"
            + json.dumps(call.get("args", {}), ensure_ascii=False, indent=2),
            show_debug,
        )


async def run_query_with_debug(
    agent: Any,
    query: str,
    config: dict[str, Any],
    show_debug: bool = False,
) -> str:
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
                    emit_debug(f"[decision] Tiếp tục search loop {loop + 1}.", show_debug)
                loop += 1
                print_tool_calls(message, loop, show_debug)
            else:
                final_message = message
                emit_debug("[decision] Dừng search và trả văn bản đích.", show_debug)

        tools_update = update.get("tools") or {}
        for message in tools_update.get("messages", []):
            if isinstance(message, ToolMessage):
                print_tool_results(message, loop, show_debug)

    if final_message is None:
        raise RuntimeError("Agent không trả về final message.")
    output = format_output(final_message.content)
    logger.info("Final result:\n%s", output)
    return output


async def run_query(agent: Any, query: str, debug: bool = False) -> str:
    logger.info("Query: %s", query)
    config = {"recursion_limit": settings.LEGAL_SEARCH_MAX_LOOPS * 2 + 4}
    return await run_query_with_debug(agent, query, config, show_debug=debug)


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
            logger.exception("Query thất bại: %s", query)
            print(f"Lỗi: {exc}", file=sys.stderr)


async def main() -> None:
    args = parse_args()
    log_path = configure_file_logging()
    print(f"Log: {log_path}")
    logger.debug(
        "Arguments: query=%r session=%s console_debug=%s",
        args.query,
        args.session,
        args.debug,
    )
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
    try:
        asyncio.run(main())
    except Exception:
        logger.exception("Legal search runner kết thúc do lỗi.")
        raise
