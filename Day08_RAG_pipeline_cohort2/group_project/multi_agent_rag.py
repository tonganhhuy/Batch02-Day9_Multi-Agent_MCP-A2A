"""Observable supervisor + workers orchestration for the Day 08 RAG pipeline."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from src.task10_generation import generate_with_citation, reorder_for_llm
from src.task9_retrieval_pipeline import retrieve


AgentName = Literal["supervisor", "retriever", "source_inspector", "answer_writer"]
RouteName = Literal["legal", "news", "mixed"]
StatusName = Literal["requested", "completed", "failed"]

WORKER_OWNED_FIELDS: dict[AgentName, set[str]] = {
    "supervisor": {"route", "route_reason", "route_plan", "chunks", "trace_insights"},
    "retriever": {"retrieval_batches"},
    "source_inspector": {"source_catalog"},
    "answer_writer": {"answer"},
}


@dataclass
class AgentMessage:
    """Minimal contract used for every supervisor-to-worker delegation."""

    trace_id: str
    message_id: str
    sender: AgentName
    recipient: AgentName
    task: str
    payload: dict[str, Any]
    status: StatusName = "requested"
    result_summary: str = ""


@dataclass
class TraceEvent:
    event_id: str
    trace_id: str
    timestamp: str
    agent: AgentName
    action: str
    status: str
    summary: str
    duration_ms: float | None = None
    message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGState:
    """Shared state. Only the supervisor applies validated worker patches."""

    trace_id: str
    query: str
    route: RouteName = "mixed"
    route_reason: str = ""
    route_plan: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    retrieval_batches: dict[str, list[dict]] = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)
    source_catalog: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    trace_insights: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _trace(
    state: RAGState,
    agent: AgentName,
    action: str,
    status: str,
    summary: str,
    duration_ms: float | None = None,
    message_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    state.trace.append(
        TraceEvent(
            event_id=str(uuid4()),
            trace_id=state.trace_id,
            timestamp=_now(),
            agent=agent,
            action=action,
            status=status,
            summary=summary,
            duration_ms=duration_ms,
            message_id=message_id,
            metadata=metadata or {},
        )
    )


def _route_query(query: str) -> tuple[RouteName, str]:
    normalized = query.casefold()
    legal_terms = ("luật", "điều", "nghị định", "thông tư", "hình phạt", "tội", "cai nghiện")
    news_terms = (
        "tin", "nghệ sĩ", "ca sĩ", "bị bắt", "khởi tố", "vụ án", "báo",
        "năm 2024", "mới nhất", "vi phạm", "cáo buộc",
    )
    legal_hits = sum(term in normalized for term in legal_terms)
    news_hits = sum(term in normalized for term in news_terms)

    if legal_hits and news_hits:
        return (
            "mixed",
            f"Câu hỏi yêu cầu đối chiếu sự kiện/nhân vật ({news_hits} tín hiệu) "
            f"với quy định pháp luật ({legal_hits} tín hiệu).",
        )
    if legal_hits > news_hits:
        return "legal", f"Phát hiện {legal_hits} tín hiệu pháp luật và {news_hits} tín hiệu tin tức."
    if news_hits > legal_hits:
        return "news", f"Phát hiện {news_hits} tín hiệu tin tức và {legal_hits} tín hiệu pháp luật."
    return "mixed", "Câu hỏi cần tổng hợp bằng chứng từ cả nguồn pháp luật và tin tức."


def _build_route_plan(route: RouteName) -> list[str]:
    if route == "legal":
        return ["retrieve:legal", "inspect_sources:mcp", "write_answer:legal"]
    if route == "news":
        return ["retrieve:news", "inspect_sources:mcp", "write_answer:news"]
    return [
        "retrieve:news",
        "bridge:news_to_legal",
        "retrieve:legal",
        "merge:evidence",
        "inspect_sources:mcp",
        "write_answer:mixed",
    ]


def _delegate(
    state: RAGState,
    recipient: AgentName,
    task: str,
    payload: dict[str, Any],
) -> AgentMessage:
    message = AgentMessage(
        trace_id=state.trace_id,
        message_id=str(uuid4()),
        sender="supervisor",
        recipient=recipient,
        task=task,
        payload=payload,
    )
    state.messages.append(message)
    _trace(
        state,
        "supervisor",
        "delegate",
        "requested",
        f"Giao '{task}' cho {recipient}.",
        message_id=message.message_id,
        metadata={"payload_keys": sorted(payload)},
    )
    return message


def _apply_worker_patch(state: RAGState, agent: AgentName, patch: dict[str, Any]) -> None:
    """Enforce field ownership before the supervisor updates shared state."""
    unauthorized = set(patch) - WORKER_OWNED_FIELDS[agent]
    if unauthorized:
        raise ValueError(f"{agent} attempted to write unauthorized fields: {sorted(unauthorized)}")
    for field_name, value in patch.items():
        setattr(state, field_name, value)


def _run_worker(state: RAGState, message: AgentMessage, worker) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        patch, summary, metadata = worker(state, message)
        _apply_worker_patch(state, message.recipient, patch)
        message.status = "completed"
        message.result_summary = summary
        _trace(
            state,
            message.recipient,
            message.task,
            "completed",
            summary,
            round((time.perf_counter() - started) * 1000, 2),
            message.message_id,
            metadata,
        )
        return patch
    except Exception as exc:
        message.status = "failed"
        message.result_summary = str(exc)
        _trace(
            state,
            message.recipient,
            message.task,
            "failed",
            f"Worker lỗi: {exc}",
            round((time.perf_counter() - started) * 1000, 2),
            message.message_id,
            {"error_type": type(exc).__name__},
        )
        raise


def retriever_worker(
    state: RAGState, message: AgentMessage
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    source_type = message.payload["source_type"]
    requested_top_k = message.payload["top_k"]
    retrieval_query = message.payload.get("retrieval_query", state.query)
    candidates = retrieve(
        retrieval_query,
        top_k=max(requested_top_k * 4, 12),
        score_threshold=message.payload["score_threshold"],
        use_reranking=message.payload["use_reranking"],
    )
    typed = [
        chunk
        for chunk in candidates
        if chunk.get("metadata", {}).get("type", "").casefold() == source_type
    ]
    # The shared retrieval pipeline searches the whole corpus. If its top list
    # misses the requested type, widen lexical search before giving up.
    if len(typed) < requested_top_k:
        try:
            from src.task6_lexical_search import lexical_search

            lexical_candidates = lexical_search(retrieval_query, top_k=80)
            typed.extend(
                chunk
                for chunk in lexical_candidates
                if chunk.get("metadata", {}).get("type", "").casefold() == source_type
                and chunk.get("content") not in {item.get("content") for item in typed}
            )
        except Exception:
            pass
    selected = typed[:requested_top_k]
    batches = dict(state.retrieval_batches)
    batches[source_type] = selected
    summary = f"Retrieval chuyên route={source_type}: {len(candidates)} candidates, chọn {len(selected)} chunks."
    return (
        {"retrieval_batches": batches},
        summary,
        {
            "source_type": source_type,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "query_enriched": retrieval_query != state.query,
        },
    )


def _build_legal_bridge_query(state: RAGState) -> str:
    """Extract observable allegation/offence phrases from news before legal lookup."""
    news_text = "\n".join(
        chunk.get("content", "") for chunk in state.retrieval_batches.get("news", [])
    )
    phrases: list[str] = []
    patterns = (
        r"(?:tội|hành vi|cáo buộc)\s+([^.;\n]{8,100})",
        r"(tổ chức sử dụng trái phép chất ma túy)",
        r"(không tố giác tội phạm)",
        r"(mua bán trái phép chất ma túy)",
    )
    for pattern in patterns:
        for match in re.findall(pattern, news_text, flags=re.IGNORECASE):
            phrase = match.strip(" _*:,")
            if phrase and phrase.casefold() not in {item.casefold() for item in phrases}:
                phrases.append(phrase)
    return f"{state.query}\nCác hành vi cần đối chiếu: {'; '.join(phrases[:4])}" if phrases else state.query


def _extract_news_allegation(state: RAGState) -> tuple[str, str]:
    """Return an observable allegation phrase and its news source."""
    candidates: list[tuple[int, str, str]] = []
    for chunk in state.retrieval_batches.get("news", []):
        content = chunk.get("content", "")
        for pattern in (
            r"(?:về tội|tội|hành vi|cáo buộc)\s+([^.;\n]{8,120})",
            r"(tổ chức sử dụng trái phép chất ma túy)",
        ):
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                allegation = match.group(1).strip(" _*:,") if match.lastindex else match.group(0)
                source = chunk.get("metadata", {}).get("source", "nguồn tin")
                normalized = allegation.casefold()
                score = (
                    5 * ("trái phép" in normalized)
                    + 3 * ("chất ma túy" in normalized)
                    + 2 * ("tổ chức sử dụng" in normalized)
                    - 2 * ("báo " in normalized or "http" in normalized)
                )
                candidates.append((score, allegation, source))
    if not candidates:
        return "", ""
    _, allegation, source = max(candidates, key=lambda item: item[0])
    return allegation, source


def _find_matching_legal_evidence(state: RAGState, allegation: str) -> dict[str, Any] | None:
    """Require strong phrase overlap before treating a legal chunk as support."""
    meaningful = {
        token
        for token in re.findall(r"\w+", allegation.casefold())
        if len(token) >= 4 and token not in {"trái", "phép", "chất"}
    }
    if not meaningful:
        return None
    for chunk in state.retrieval_batches.get("legal", []):
        content = chunk.get("content", "").casefold()
        overlap = sum(token in content for token in meaningful)
        if overlap >= max(3, len(meaningful) - 1) and ("điều " in content or "tội " in content):
            return chunk
    return None


def _call_mcp_source_catalog(
    source_names: list[str], max_attempts: int = 2
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Discover then call the MCP tool, with retry and graceful fallback."""
    server = Path(__file__).with_name("mcp_source_server.py")
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "inspect_source_catalog",
                        "arguments": {"source_names": source_names},
                    },
                },
            ]
            completed = subprocess.run(
                [sys.executable, str(server)],
                input="\n".join(json.dumps(item, ensure_ascii=False) for item in requests) + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                check=True,
            )
            responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            tools = responses[1]["result"]["tools"]
            discovered = [tool["name"] for tool in tools]
            if "inspect_source_catalog" not in discovered:
                raise RuntimeError("Required MCP tool was not discovered")
            catalog = responses[2]["result"]["structuredContent"]
            return catalog, {"discovered_tools": discovered, "attempts": attempt, "fallback": False}
        except Exception as exc:
            last_error = str(exc)

    fallback = {
        "matched_count": 0,
        "available_types": [],
        "sources": [],
        "verification_status": "unavailable",
    }
    return fallback, {
        "discovered_tools": [],
        "attempts": max_attempts,
        "fallback": True,
        "error": last_error,
    }


def source_inspector_worker(
    state: RAGState, message: AgentMessage
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    source_names = sorted(
        {
            chunk.get("metadata", {}).get("source", "")
            for chunk in state.chunks
            if chunk.get("metadata", {}).get("source")
        }
    )
    catalog, call_metadata = _call_mcp_source_catalog(source_names)
    catalog["verification_status"] = "verified" if catalog.get("matched_count", 0) else "unverified"
    summary = (
        f"MCP discovery thành công; xác minh {catalog.get('matched_count', 0)}/{len(source_names)} nguồn."
        if not call_metadata["fallback"]
        else "MCP không khả dụng sau retry; tiếp tục với trạng thái nguồn chưa xác minh."
    )
    return {"source_catalog": catalog}, summary, call_metadata


def answer_writer_worker(
    state: RAGState, message: AgentMessage
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    allegation, news_source = _extract_news_allegation(state)
    legal_evidence = _find_matching_legal_evidence(state, allegation) if allegation else None

    if state.route == "mixed" and allegation and not legal_evidence:
        answer = (
            f"Nguồn tin cho biết cáo buộc/hành vi liên quan là **{allegation}** "
            f"[{news_source}].\n\n"
            "Tuy nhiên, các đoạn văn bản pháp luật hiện có trong kho chưa chứa phần điều luật "
            "khớp trực tiếp với hành vi này. Vì vậy hệ thống **chưa đủ căn cứ để xác minh "
            "chính xác điều, khoản hoặc khung hình phạt** từ nguồn legal hiện tại. "
            "Cần bổ sung toàn văn Bộ luật Hình sự 2015 để kết luận."
        )
        answer_mode = "cross_source_evidence_gap"
    else:
        result = generate_with_citation(
            state.query,
            top_k=message.payload["top_k"],
            chunks=state.chunks,
        )
        answer = result.get("answer") or "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        answer_mode = "rag_generation"
    verification = state.source_catalog.get("verification_status", "unverified")
    verified_count = state.source_catalog.get("matched_count", 0)
    answer += f"\n\n_Trạng thái nguồn qua MCP: **{verification}** ({verified_count} nguồn khớp catalog)._"
    return (
        {"answer": answer},
        f"Tạo câu trả lời route={state.route} từ {len(state.chunks)} chunks; source={verification}.",
        {
            "chunk_count": len(state.chunks),
            "verification_status": verification,
            "answer_mode": answer_mode,
            "news_allegation_found": bool(allegation),
            "matching_legal_evidence_found": bool(legal_evidence),
        },
    )


def _merge_retrieval_batches(state: RAGState, top_k: int) -> None:
    merged: list[dict] = []
    seen: set[str] = set()
    merge_order = ("news", "legal") if state.route == "mixed" else (state.route,)
    for source_type in merge_order:
        for chunk in state.retrieval_batches.get(source_type, []):
            key = chunk.get("content", "")
            if key and key not in seen:
                seen.add(key)
                merged.append(chunk)
    state.chunks = reorder_for_llm(merged[:top_k])
    _trace(
        state,
        "supervisor",
        "merge_evidence",
        "completed",
        f"Merge {len(state.retrieval_batches)} retrieval batches thành {len(state.chunks)} chunks.",
        metadata={"batch_sizes": {name: len(items) for name, items in state.retrieval_batches.items()}},
    )


def _build_trace_insights(state: RAGState) -> dict[str, Any]:
    timed = [event for event in state.trace if event.duration_ms is not None]
    slowest = max(timed, key=lambda event: event.duration_ms, default=None)
    return {
        "route_path": " -> ".join(state.route_plan),
        "total_worker_duration_ms": round(sum(event.duration_ms or 0 for event in timed), 2),
        "slowest_agent": slowest.agent if slowest else "none",
        "slowest_action": slowest.action if slowest else "none",
        "slowest_duration_ms": slowest.duration_ms if slowest else 0,
        "failed_events": sum(event.status == "failed" for event in state.trace),
        "mcp_verification": state.source_catalog.get("verification_status", "unverified"),
        "retrieval_batch_sizes": {
            name: len(items) for name, items in state.retrieval_batches.items()
        },
    }


def run_multi_agent_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    use_reranking: bool = True,
) -> RAGState:
    """Run conditional supervised workflow and return answer plus observable trace."""
    state = RAGState(
        trace_id=str(uuid4()),
        query=query,
        config={
            "top_k": top_k,
            "score_threshold": score_threshold,
            "use_reranking": use_reranking,
        },
    )
    state.route, state.route_reason = _route_query(query)
    state.route_plan = _build_route_plan(state.route)
    _trace(
        state,
        "supervisor",
        "route",
        "completed",
        f"Chọn route={state.route}. {state.route_reason}",
        metadata={"conditional_plan": state.route_plan},
    )

    source_types = ["news", "legal"] if state.route == "mixed" else [state.route]
    for source_type in source_types:
        retrieval_query = (
            _build_legal_bridge_query(state)
            if state.route == "mixed" and source_type == "legal"
            else state.query
        )
        if retrieval_query != state.query:
            _trace(
                state,
                "supervisor",
                "bridge_news_to_legal",
                "completed",
                "Dùng cáo buộc/hành vi quan sát được từ nguồn news để truy vấn điều luật liên quan.",
                metadata={"enriched_query": retrieval_query},
            )
        message = _delegate(
            state,
            "retriever",
            f"retrieve_{source_type}_evidence",
            {**state.config, "source_type": source_type, "retrieval_query": retrieval_query},
        )
        _run_worker(state, message, retriever_worker)
    _merge_retrieval_batches(state, top_k)

    source_message = _delegate(
        state,
        "source_inspector",
        "discover_and_verify_sources_via_mcp",
        {"source_count": len(state.chunks), "required_tool": "inspect_source_catalog"},
    )
    _run_worker(state, source_message, source_inspector_worker)

    answer_message = _delegate(
        state,
        "answer_writer",
        "write_verified_cited_answer",
        {"top_k": top_k, "route": state.route, "verification_required": True},
    )
    _run_worker(state, answer_message, answer_writer_worker)

    state.trace_insights = _build_trace_insights(state)
    _trace(
        state,
        "supervisor",
        "finish",
        "completed",
        "Đã tổng hợp câu trả lời, source verification và actionable trace insights.",
        metadata=state.trace_insights,
    )
    return state
