import pytest

from group_project.mcp_source_server import handle_request
from group_project.multi_agent_rag import (
    RAGState,
    _apply_worker_patch,
    _build_route_plan,
    _extract_news_allegation,
    _route_query,
)


def test_supervisor_routes_and_builds_conditional_plans():
    assert _route_query("Điều 248 quy định hình phạt thế nào?")[0] == "legal"
    assert _route_query("Tin nghệ sĩ bị bắt năm 2024")[0] == "news"
    assert _build_route_plan("legal") != _build_route_plan("news")
    mixed_plan = _build_route_plan("mixed")
    assert "retrieve:legal" in mixed_plan
    assert "retrieve:news" in mixed_plan
    assert "merge:evidence" in mixed_plan
    assert _route_query(
        "Cho tôi biết ca sĩ Miu Lê đã vi phạm điều gì trong bộ luật 2015 sửa đổi 2017"
    )[0] == "mixed"
    assert mixed_plan.index("retrieve:news") < mixed_plan.index("retrieve:legal")


def test_state_ownership_rejects_unauthorized_worker_write():
    state = RAGState(trace_id="trace-1", query="test")
    with pytest.raises(ValueError, match="unauthorized"):
        _apply_worker_patch(state, "answer_writer", {"chunks": []})


def test_mcp_server_exposes_discoverable_source_catalog_tool():
    response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = response["result"]["tools"]
    assert tools[0]["name"] == "inspect_source_catalog"
    assert tools[0]["inputSchema"]["properties"]["source_names"]["type"] == "array"


def test_mcp_server_validates_tool_arguments():
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "inspect_source_catalog",
                "arguments": {"source_names": "not-an-array"},
            },
        }
    )
    assert response["error"]["code"] == -32602


def test_mcp_server_reads_catalog():
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "inspect_source_catalog", "arguments": {}},
        }
    )
    result = response["result"]["structuredContent"]
    assert result["matched_count"] > 0
    assert "legal" in result["available_types"]


def test_extracts_news_allegation_for_cross_source_question():
    state = RAGState(trace_id="trace-1", query="Miu Lê vi phạm điều gì?")
    state.retrieval_batches["news"] = [
        {
            "content": "Ca sĩ Miu Lê bị bắt tạm giam về tội Tổ chức sử dụng trái phép chất ma túy.",
            "metadata": {"source": "article_01.md", "type": "news"},
        }
    ]
    allegation, source = _extract_news_allegation(state)
    assert "Tổ chức sử dụng trái phép chất ma túy" in allegation
    assert source == "article_01.md"
