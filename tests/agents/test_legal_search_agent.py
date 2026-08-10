import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

module_path = Path(__file__).parents[2] / "src" / "agents" / "legal_search_agent.py"
spec = importlib.util.spec_from_file_location("legal_search_agent_under_test", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_agent_exposes_only_candidate_and_document_verification_tools() -> None:
    assert [tool.name for tool in module.tools] == [
        "search_legal_documents",
        "search_law_terms",
        "search_law_terms_in_document",
    ]
    assert "doc_id" not in module.tools[1].args
    assert module.tools[2].args["doc_id"].get("default") is None
    assert "default" not in module.tools[2].args["doc_id"]


def test_tool_descriptions_define_staged_search() -> None:
    descriptions = {tool.name: tool.description for tool in module.tools}

    assert "không nhận doc_id" in descriptions["search_law_terms"]
    assert "vòng sau" in descriptions["search_law_terms_in_document"]
    assert "không được tự đoán" in descriptions["search_law_terms_in_document"]


def test_instructions_disable_title_and_require_retrieval_only_output() -> None:
    assert "Không sử dụng search_law_titles" in module.instructions
    assert "search_law_terms là tìm kiếm toàn cục" in module.instructions
    assert "không có filter doc_id" in module.instructions
    assert "Không gọi tìm candidate và kiểm chứng trong cùng một vòng" in module.instructions
    assert '{"documents": [' in module.instructions


def test_instructions_require_evidence_for_every_search_intent() -> None:
    assert "tách câu hỏi" in module.instructions
    assert "mỗi tool call" in module.instructions
    assert "cho một\n  intent" in module.instructions
    assert "Phải có kết quả tool phù hợp cho từng intent trước khi dừng" in module.instructions
    assert "hồ sơ thủ tục đăng ký" in module.instructions
    assert 'tạm trú" và' in module.instructions
    assert "trình tự thủ tục khai báo tạm vắng" in module.instructions
    assert "documents rỗng" in module.instructions


def test_known_candidate_ids_reads_es_and_global_term_results() -> None:
    state = {
        "messages": [
            ToolMessage(
                content=json.dumps([{"doc_info": {"id": "173920"}}]),
                name="search_legal_documents",
                tool_call_id="es-1",
            ),
            ToolMessage(
                content=json.dumps([{"doc_id": 170620}]),
                name="search_law_terms",
                tool_call_id="term-1",
            ),
            AIMessage(content="pending"),
        ]
    }

    assert module.known_candidate_ids(state) == {"173920", "170620"}


@pytest.mark.asyncio
async def test_call_tools_rejects_unknown_document_id(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_call = {
        "name": "search_law_terms_in_document",
        "args": {"query": "x", "doc_id": 999},
        "id": "verify-1",
    }
    tool_node_call = AsyncMock()
    monkeypatch.setattr(module.tool_node, "ainvoke", tool_node_call)

    result = await module.call_tools(
        {
            "messages": [
                HumanMessage(content="question"),
                AIMessage(content="", tool_calls=[tool_call]),
            ]
        },
        {},
    )

    tool_node_call.assert_not_awaited()
    assert result["messages"][0].status == "error"
    assert "doc_id phải xuất hiện" in result["messages"][0].content


@pytest.mark.asyncio
async def test_call_tools_accepts_candidate_from_previous_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_node_call = AsyncMock(return_value={"messages": []})
    monkeypatch.setattr(module.tool_node, "ainvoke", tool_node_call)
    candidate_result = ToolMessage(
        content=json.dumps([{"doc_info": {"id": "173920"}}]),
        name="search_legal_documents",
        tool_call_id="es-1",
    )

    await module.call_tools(
        {
            "messages": [
                candidate_result,
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_law_terms_in_document",
                            "args": {"query": "x", "doc_id": 173920},
                            "id": "verify-1",
                        }
                    ],
                ),
            ]
        },
        {},
    )

    tool_node_call.assert_awaited_once()


def test_normalize_final_response_flattens_gemini_text_blocks() -> None:
    response = AIMessage(content=[{"type": "text", "text": '{"documents": []}'}])

    normalized = module.normalize_final_response(response)

    assert normalized.content == '{"documents": []}'


def test_route_after_model_routes_tool_calls() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_legal_documents", "args": {"query": "x"}, "id": "1"}],
            )
        ]
    }

    assert module.route_after_model(state) == "tools"
    assert module.route_after_model({"messages": [AIMessage(content="done")]}) == "done"


@pytest.mark.asyncio
async def test_call_model_disables_tools_at_max_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content="final"))
    monkeypatch.setattr(module, "get_legal_search_model", lambda: model)
    monkeypatch.setattr(module.settings, "LEGAL_SEARCH_MAX_LOOPS", 2)

    result = await module.call_model(
        {"messages": [HumanMessage(content="question")], "search_loops": 2},
        {},
    )

    model.bind_tools.assert_not_called()
    assert result["messages"][0].content == "final"


@pytest.mark.asyncio
async def test_call_model_enables_tools_below_max_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    bound_model = Mock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="searching"))
    model = Mock()
    model.bind_tools.return_value = bound_model
    monkeypatch.setattr(module, "get_legal_search_model", lambda: model)
    monkeypatch.setattr(module.settings, "LEGAL_SEARCH_MAX_LOOPS", 2)

    await module.call_model(
        {"messages": [HumanMessage(content="question")], "search_loops": 1},
        {},
    )

    model.bind_tools.assert_called_once_with(module.tools)
