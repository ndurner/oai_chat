---
title: OAI Chat
emoji: 🤖
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
pinned: false
license: mit
---

# OAI Chat

Chat interface based on OpenAI transformer models. \
Features:
 * Image upload (support for vision via gpt-4-vision)
 * Word file (DOCX) upload
 * PDF file support (via image rendering & GPT-4V)
 * Plaintext file upload
 * chat history download
 * file download
   * example: download an ICS calendar file the model has created for you
* streaming chat
* image generation (via DALL-E 3)
* remote MCP server support via configurable registry
* optional UnrestrictedPython execution when `CODE_EXEC_UNRESTRICTED_PYTHON=1`

The MCP registry is looked up in the following order:
1. `$OAI_CHAT_MCP_REGISTRY` if set
2. `mcp_registry.json` in this repository
3. `~/.oai_chat/mcp_registry.json`

See `mcp_registry.sample.json` for an example configuration.
Headers and query parameters may reference environment variables using the `env:` prefix.
Use `"allowed_tools": ["*"]` to permit all tools from a server.
When an MCP tool requires approval, the assistant will notify you in chat.
Reply with `y` to approve or `n` to deny the request, optionally adding a comment after the `y` or `n`.
