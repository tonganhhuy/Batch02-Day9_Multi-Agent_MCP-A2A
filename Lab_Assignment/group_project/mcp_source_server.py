"""Minimal MCP-compatible stdio server exposing the local source catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"


def inspect_source_catalog(source_names: list[str] | None = None) -> dict:
    """Return metadata for indexed source files without exposing file contents."""
    requested = {Path(name).name for name in (source_names or [])}
    sources = []

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if requested and path.name not in requested:
            continue
        sources.append(
            {
                "name": path.name,
                "type": path.parent.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "matched_count": len(sources),
        "available_types": sorted({source["type"] for source in sources}),
        "sources": sources,
    }


def handle_request(request: dict) -> dict:
    request_id = request.get("id")
    method = request.get("method")

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "druglaw-source-catalog", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "inspect_source_catalog",
                    "description": "Inspect metadata for local legal and news sources.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source_names": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                }
            ]
        }
    elif method == "tools/call":
        params = request.get("params", {})
        if params.get("name") != "inspect_source_catalog":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Unknown tool"},
            }
        arguments = params.get("arguments", {})
        source_names = arguments.get("source_names")
        if source_names is not None and (
            not isinstance(source_names, list)
            or not all(isinstance(name, str) for name in source_names)
        ):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": "source_names must be an array of strings"},
            }
        catalog = inspect_source_catalog(source_names)
        result = {
            "content": [{"type": "text", "text": json.dumps(catalog, ensure_ascii=False)}],
            "structuredContent": catalog,
            "isError": False,
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(exc)},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
