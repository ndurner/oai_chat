import os
import json
from urllib.parse import urlencode


_SEARCH_PATHS = [
    os.getenv("OAI_CHAT_MCP_REGISTRY"),
    os.path.join(os.path.dirname(__file__), "mcp_registry.json"),
    os.path.expanduser("~/.oai_chat/mcp_registry.json"),
]


def _merge_defaults(reg: dict) -> list[dict]:
    defaults = reg.get("defaults", {})
    servers = []
    for entry in reg.get("servers", []):
        if entry.get("url"):
            merged = dict(defaults)
            merged.update(entry)
            servers.append(merged)
        else:
            # Local MCPs not yet supported
            pass
    return servers


def load_registry() -> list[dict]:
    for path in _SEARCH_PATHS:
        if path and os.path.exists(path):
            with open(path) as f:
                return _merge_defaults(json.load(f))
    return []


def env_subst(values: dict, kind: str) -> dict:
    out = {}
    for k, v in values.items():
        if isinstance(v, str) and v.startswith("env:"):
            env_name = v[4:]
            if env_name not in os.environ:
                raise RuntimeError(f"Missing env var {env_name} for MCP {kind} {k}")
            out[k] = os.environ[env_name]
        else:
            out[k] = v
    return out


def to_openai_tool(entry: dict) -> dict:
    server_url = entry["url"]
    if "query_params" in entry:
        qp = urlencode(env_subst(entry["query_params"], "query parameter"))
        if "?" in server_url:
            server_url += "&" + qp
        else:
            server_url += "?" + qp
    tool = {
        "type": "mcp",
        "server_label": entry.get("server_label", entry["name"]),
        "server_url": server_url,
        "headers": env_subst(entry.get("headers", {}), "header"),
    }
    if "allowed_tools" in entry:
        allowed = entry["allowed_tools"]
        if not (len(allowed) == 1 and allowed[0] == "*"):
            tool["allowed_tools"] = allowed
    if "require_approval" in entry:
        tool["require_approval"] = entry["require_approval"]
    return tool
