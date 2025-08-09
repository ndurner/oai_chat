---
title: OAI Chat
emoji: 🤖
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
license: mit
---

# OAI Chat

Chat interface for LLMs via the legacy OpenAI-compatible Chat Completions API. OpenAI models are best used via the OpenAI Responses API (`main` branch), this exists for compatibility.

Features:
 * support for OpenAI GPT-OSS models via Amazon Bedrock
 * Image upload (support for vision via gpt-4-vision)
 * Word file (DOCX) upload
 * PDF file support (via image rendering & GPT-4V)
 * Plaintext file upload
 * chat history download
 * file download
   * example: download an ICS calendar file the model has created for you
* streaming chat
* image generation (via DALL-E 3)

**Quickstart**
- Create a virtualenv and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run the app: `python app.py`.
- A Gradio UI launches locally; open the provided URL.

**Configuration**
- OpenAI: Provide your key in the UI or set `OPENAI_API_KEY`.
- AWS-compatible endpoint: Set `AWS_OPENAI_ENDPOINT` and `AWS_BEARER_TOKEN_BEDROCK` to use an OpenAI-compatible Bedrock endpoint. The UI will switch the key label and default models accordingly.
- .env support: The app auto-loads a local `.env` if present (via `python-dotenv`). Example:
  - `OPENAI_API_KEY=sk-...`
  - or
  - `AWS_OPENAI_ENDPOINT=https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1`
  - `AWS_BEARER_TOKEN_BEDROCK=A...`

**Models**
- OpenAI: common options like `gpt-4o`, `gpt-4-turbo`, `o1`, `o3-mini`, plus `whisper` and `dall-e-3`.
- AWS-compatible: `openai.gpt-oss-20b-1:0`, `openai.gpt-oss-120b-1:0`.
- The model dropdown allows custom entries if your provider supports more.

**Notes**
- Credentials check: On first use, the app performs a lightweight check to validate your credentials against the selected provider.
- Requirements now include `python-dotenv` so `.env` files are read automatically at startup.
