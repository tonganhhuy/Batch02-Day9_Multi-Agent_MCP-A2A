# Multi-Agent RAG Architecture

## Flow

```text
User
  -> Supervisor (route: legal | news | mixed)
      -> Retriever Worker (hybrid retrieval + route filtering)
      -> Source Inspector Worker (MCP tool: inspect_source_catalog)
      -> Answer Writer Worker (generation + citation)
  -> Final answer + sources + observable trace
```

The supervisor builds a conditional execution plan. Legal and news routes
delegate one specialized retrieval task. A mixed route delegates both legal and
news retrieval tasks and adds an explicit supervisor merge edge.

```text
legal -> retrieve:legal -> inspect_sources:mcp -> write_answer:legal
news  -> retrieve:news  -> inspect_sources:mcp -> write_answer:news
mixed -> retrieve:legal -> retrieve:news -> merge:evidence
      -> inspect_sources:mcp -> write_answer:mixed
```

## Shared State

`RAGState` contains:

- `trace_id`, `query`, `route`, `route_reason`, and runtime `config`
- retrieved `chunks`, MCP `source_catalog`, and final `answer`
- `messages`: every supervisor-to-worker delegation
- `trace`: observable agent actions, results, status, and duration
- `trace_insights`: slowest action, failures, route path, batch sizes, MCP status

Workers return state patches instead of mutating shared state directly. The
supervisor validates every patch against `WORKER_OWNED_FIELDS` before applying
it, making field ownership explicit and enforceable.

## Minimal Message Contract

```json
{
  "trace_id": "uuid",
  "message_id": "uuid",
  "sender": "supervisor",
  "recipient": "retriever | source_inspector | answer_writer",
  "task": "task_name",
  "payload": {},
  "status": "requested | completed | failed"
}
```

## MCP Capability

The Source Inspector worker starts `mcp_source_server.py` over stdio, discovers
available tools using `tools/list`, validates that `inspect_source_catalog`
exists, then calls it. The client retries failures and records a graceful
fallback status. The answer includes the MCP source-verification status, so the
capability affects final output instead of acting as a display-only call.

## Demo

```powershell
streamlit run group_project/app.py
```

The UI shows the final answer, selected route, source documents, message
contract, MCP result, and observable reasoning flow. The displayed flow is an
action/result trace for debugging, not private model chain-of-thought.

`Fast mode` is enabled by default. It uses the cached BM25 index scoped by route
and skips loading semantic/cross-encoder models for every worker call. Disable
it in the sidebar when a slower full hybrid retrieval run is needed.

## Rubric Evidence

| Criterion | Evidence |
|---|---|
| Clear roles | Supervisor owns routing/state updates; Retriever, Source Inspector, and Answer Writer have non-overlapping capabilities. |
| MCP integration | Client performs `initialize -> tools/list -> tools/call`, validates discovery, retries failures, and records fallback status. |
| Shared state | `RAGState` is typed; `WORKER_OWNED_FIELDS` prevents unauthorized worker writes. |
| Trace quality | Events contain IDs, timestamps, agent, action, status, summary, duration, message link, and metadata. `trace_insights` identifies slowest action, failures, route path, and retrieval batch sizes. |
| Routing logic | Route reason is recorded and each route produces a different conditional execution plan. Mixed route explicitly fans out and merges evidence. |

### Anti-patterns intentionally avoided

- Workers do not route or call each other; only the supervisor delegates work.
- Workers do not mutate arbitrary shared-state fields; they return validated patches.
- The MCP tool is a stateless capability, not disguised as an agent.
- MCP failure does not crash the whole workflow; it becomes an observable
  `unverified` result.
- The UI exposes action/result traces, not private model chain-of-thought.
