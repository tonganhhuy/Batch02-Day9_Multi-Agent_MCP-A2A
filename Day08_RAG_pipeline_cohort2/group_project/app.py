from pathlib import Path
import sys

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from group_project.multi_agent_rag import run_multi_agent_rag


st.set_page_config(page_title="Multi-Agent DrugLaw RAG", page_icon="🔎", layout="wide")

st.markdown(
    """
<style>
    .stApp { background-color: #0d0f12; color: #e2e8f0; }
    .main-header { color: #60a5fa; font-size: 2.6rem; font-weight: 800; }
    .card { background: #111827; padding: 1rem; border: 1px solid #334155;
            border-radius: .7rem; margin-bottom: .7rem; }
    .trace { border-left: 4px solid #a78bfa; }
    .source { border-left: 4px solid #10b981; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Cấu hình hệ thống")
    use_reranking = st.toggle("Kích hoạt reranking", value=True)
    score_threshold = st.slider("Ngưỡng retrieval", 0.0, 1.0, 0.3, 0.05)
    top_k = st.slider("Số context", 1, 10, 5)
    st.divider()
    st.subheader("Agent capabilities")
    st.code(
        "Supervisor: route + delegate\n"
        "Retriever: hybrid RAG\n"
        "Source Inspector: MCP tool\n"
        "Answer Writer: cited answer",
        language=None,
    )

st.markdown("<div class='main-header'>Multi-Agent DrugLaw RAG</div>", unsafe_allow_html=True)
st.caption("Supervisor + 3 workers, có route rõ, MCP capability và observable trace.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Nhập câu hỏi về luật hoặc tin tức ma túy...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Supervisor đang điều phối các worker..."):
            state = run_multi_agent_rag(
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
            )
        st.markdown(state.answer)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Route", state.route)
        col2.metric("Workers", 3)
        col3.metric("Trace events", len(state.trace))
        col4.metric("MCP verification", state.trace_insights["mcp_verification"])
        st.caption(f"Trace ID: `{state.trace_id}` | {state.route_reason}")

        with st.expander("Conditional route plan", expanded=True):
            st.code(" -> ".join(state.route_plan), language=None)

        with st.expander("Reasoning flow quan sát được", expanded=True):
            st.info(
                "Đây là các quyết định, hành động và kết quả có thể quan sát để debug; "
                "không phải chain-of-thought nội bộ của mô hình."
            )
            for event in state.trace:
                duration = f" · {event.duration_ms:.2f} ms" if event.duration_ms is not None else ""
                st.markdown(
                    f"<div class='card trace'><strong>{event.agent}</strong> → "
                    f"<code>{event.action}</code> [{event.status}]{duration}<br>"
                    f"{event.summary}</div>",
                    unsafe_allow_html=True,
                )

        with st.expander("Actionable trace insights", expanded=True):
            st.json(state.trace_insights)

        with st.expander("Message contract supervisor ↔ workers"):
            st.json([message.__dict__ for message in state.messages])

        with st.expander("MCP capability result"):
            st.json(state.source_catalog)

        with st.expander(f"Nguồn tài liệu đã dùng ({len(state.chunks)})"):
            for index, chunk in enumerate(state.chunks, 1):
                metadata = chunk.get("metadata", {})
                st.markdown(
                    f"<div class='card source'><strong>{index}. "
                    f"{metadata.get('source', 'Chưa rõ')}</strong> · "
                    f"{metadata.get('type', 'unknown')} · score={chunk.get('score', 0):.3f}<br>"
                    f"{chunk.get('content', '')}</div>",
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({"role": "assistant", "content": state.answer})
