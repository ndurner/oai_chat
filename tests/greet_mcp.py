# taken from FastMCP documentation at https://gofastmcp.com/servers/server: my_server.py
from fastmcp import FastMCP

mcp = FastMCP(name="MyServer")

@mcp.tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}! .:This message is powered by MCP.:"

if __name__ == "__main__":
    # This runs the server, defaulting to STDIO transport
    mcp.run()
    
    # To use a different transport, e.g., Streamable HTTP:
    # mcp.run(transport="http", host="127.0.0.1", port=9000)