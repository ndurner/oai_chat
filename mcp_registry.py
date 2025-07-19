import os
import json
import asyncio
import logging
from urllib.parse import urlencode
from typing import Dict, List, Any, Optional, Union

try:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport, PythonStdioTransport
except ImportError:
    logging.warning("FastMCP library not installed. Local MCP servers will not be available.")
    Client = None
    StdioTransport = None

# Global dictionary to store local MCP clients
local_mcp_clients = {}
local_mcp_tools_cache = {}


_SEARCH_PATHS = [
    os.getenv("OAI_CHAT_MCP_REGISTRY"),
    os.path.join(os.path.dirname(__file__), "mcp_registry.json"),
    os.path.expanduser("~/.oai_chat/mcp_registry.json"),
]

async def log(msg):
    print("[MCP SERVER]", msg.data, flush=True)


def _merge_defaults(reg: dict) -> list[dict]:
    defaults = reg.get("defaults", {})
    servers = []
    for entry in reg.get("servers", []):
        merged = dict(defaults)
        merged.update(entry)
        servers.append(merged)
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


def is_local_mcp(entry: dict) -> bool:
    """Check if an MCP entry is a local MCP server"""
    return "command" in entry and "args" in entry

async def start_local_mcp_client(entry: dict) -> Optional[Client]:
    """Start a local MCP client for a given entry"""
    if Client is None or StdioTransport is None:
        logging.error("FastMCP library not installed. Cannot start local MCP client.")
        return None
        
    try:
        name = entry["name"]
        command = entry["command"]
        args = entry["args"]
        
        # Prepare environment variables
        env_vars = {}
        if "env" in entry:
            env_vars = env_subst(entry["env"], "environment variable")
        
        # Create transport with environment variables
        transport = StdioTransport(
            command=command,
            args=args,
            env=env_vars if env_vars else None
        )
        
        # Create client with the transport
        client = Client(transport, log_handler=log)
        
        # Store the client in the global dictionary
        local_mcp_clients[name] = client
        
        return client
    except Exception as e:
        logging.error(f"Failed to start local MCP client: {str(e)}")
        return None

async def get_local_mcp_tools(entry: dict) -> List[Dict[str, Any]]:
    """Get available tools from a local MCP server"""
    name = entry["name"]
    
    # Check if we have cached tools for this server
    if name in local_mcp_tools_cache:
        return local_mcp_tools_cache[name]
        
    # Check if client exists or create a new one
    client = local_mcp_clients.get(name)
    if client is None:
        client = await start_local_mcp_client(entry)
        if client is None:
            return []
    
    try:
        # Use client in async context manager
        async with client:
            # List available tools
            tools = await client.list_tools()
            # Cache the tools
            local_mcp_tools_cache[name] = tools
            return tools
    except Exception as e:
        logging.error(f"Failed to list tools from local MCP server: {str(e)}")
        return []

async def call_local_mcp_tool(entry: dict, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool on a local MCP server"""
    name = entry["name"]
    
    # Check if client exists or create a new one
    client = local_mcp_clients.get(name)
    if client is None:
        client = await start_local_mcp_client(entry)
        if client is None:
            return {"error": "Failed to connect to local MCP server"}
    
    try:
        # Use client in async context manager
        async with client:
            if not client.is_connected():
                logging.warning("MCP server not connected")

            # Call the tool
            result = await client.call_tool(tool_name, arguments)
            return result
    except Exception as e:
        logging.error(f"Failed to call tool on local MCP server: {str(e)}")
        return {"error": str(e)}

async def shutdown_local_mcp_clients():
    """Shutdown all local MCP clients"""
    for name, client in local_mcp_clients.items():
        try:
            await client.close()
        except Exception as e:
            logging.error(f"Failed to close local MCP client {name}: {str(e)}")
    local_mcp_clients.clear()
    local_mcp_tools_cache.clear()

def to_openai_tool(entry: dict) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert an MCP entry to an OpenAI tool definition(s)"""
    # For remote MCP servers, use the standard "mcp" type
    if "url" in entry:
        tool = {
            "type": "mcp",
            "server_label": entry.get("server_label", entry["name"]),
        }
        
        server_url = entry["url"]
        if "query_params" in entry:
            qp = urlencode(env_subst(entry["query_params"], "query parameter"))
            if "?" in server_url:
                server_url += "&" + qp
            else:
                server_url += "?" + qp
        tool["server_url"] = server_url
        tool["headers"] = env_subst(entry.get("headers", {}), "header")
    if "allowed_tools" in entry:
        allowed = entry["allowed_tools"]
        if not (len(allowed) == 1 and allowed[0] == "*"):
            tool["allowed_tools"] = allowed
    if "require_approval" in entry:
        tool["require_approval"] = entry["require_approval"]
    return tool

# Global mapping to track function names back to their MCP servers and tool names
function_to_mcp_map = {}

# Cache for local MCP tools
local_mcp_tool_cache = {}

# Helper function to create a function tool definition for a local MCP tool
def create_function_tool_for_local_mcp_tool(server_name: str, tool_name: str, tool_obj) -> Dict[str, Any]:
    """Create an OpenAI function tool definition for a local MCP tool"""
    function_name = f"{server_name}_{tool_name}"
    
    # Save the mapping for later lookup during function call
    function_to_mcp_map[function_name] = {
        "server_name": server_name,
        "tool_name": tool_name
    }
    
    # Handle FastMCP Tool object format (based on observed structure)
    description = getattr(tool_obj, 'description', f"Tool {tool_name} from {server_name} MCP server")
    parameters = getattr(tool_obj, 'inputSchema', {"type": "object", "properties": {}})
    
    return {
        "type": "function",
        "name": function_name,
        "description": description,
        "parameters": parameters
    }

async def get_tools_for_server(entry: dict) -> List[Dict[str, Any]]:
    """Get all tools for a given server entry (local or remote)
    For remote servers, it returns a single MCP tool.
    For local servers, it returns multiple function tools (one for each MCP tool).
    """
    if is_local_mcp(entry):
        server_name = entry["name"]
        # Try to get tools from cache first
        if server_name in local_mcp_tool_cache:
            mcp_tools = local_mcp_tool_cache[server_name]
        else:
            try:
                mcp_tools = await get_local_mcp_tools(entry)
                local_mcp_tool_cache[server_name] = mcp_tools
            except Exception as e:
                logging.error(f"Error getting tools from local MCP server {server_name}: {str(e)}")
                mcp_tools = []
        result = []
        for tool_obj in mcp_tools:
            tool_name = getattr(tool_obj, 'name', None)
            if tool_name:
                function_tool = create_function_tool_for_local_mcp_tool(server_name, tool_name, tool_obj)
                result.append(function_tool)
        return result
    else:
        tool = to_openai_tool(entry)
        if isinstance(tool, list):
            return tool
        else:
            return [tool]
