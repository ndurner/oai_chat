import gradio as gr
import base64
import os
from pathlib import Path
from openai import OpenAI
import json
from PIL import Image
import io
from settings_mgr import generate_download_settings_js, generate_upload_settings_js
from chat_export import import_history, get_export_js
from mcp_registry import load_registry, to_openai_tool
from gradio.components.base import Component
from types import SimpleNamespace

from doc2json import process_docx
from code_exec import eval_script

dump_controls = False
log_to_console = False

mcp_servers = load_registry()
pending_mcp_request = None

def load_openai_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = Path(".env")
    if env_path.is_file():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, sep, v = line.partition("=")
                if k == "OPENAI_API_KEY" and sep:
                    return v.strip().strip('"').strip("'")
    return ""

def encode_image(image_data):
    """Generates a prefix for image base64 data in the required format for the
    four known image formats: png, jpeg, gif, and webp.

    Args:
    image_data: The image data, encoded in base64.

    Returns:
    A string containing the prefix.
    """

    # Get the first few bytes of the image data.
    magic_number = image_data[:4]
  
    # Check the magic number to determine the image type.
    if magic_number.startswith(b'\x89PNG'):
        image_type = 'png'
    elif magic_number.startswith(b'\xFF\xD8'):
        image_type = 'jpeg'
    elif magic_number.startswith(b'GIF89a'):
        image_type = 'gif'
    elif magic_number.startswith(b'RIFF'):
        if image_data[8:12] == b'WEBP':
            image_type = 'webp'
        else:
            # Unknown image type.
            raise Exception("Unknown image type")
    else:
        # Unknown image type.
        raise Exception("Unknown image type")

    return f"data:image/{image_type};base64,{base64.b64encode(image_data).decode('utf-8')}"

def process_pdf(pdf_fn: str):
    with open(pdf_fn, "rb") as pdf_file:
        base64_string = base64.b64encode(pdf_file.read()).decode("utf-8")
    return [{"type": "input_file", "filename": os.path.basename(pdf_fn), 
        "file_data": f"data:application/pdf;base64,{base64_string}"}]

def encode_file(fn: str) -> list:
    user_msg_parts = []

    if fn.endswith(".docx"):
        user_msg_parts.append({"type": "input_text", "text": process_docx(fn)})
    elif fn.endswith(".pdf"):
        user_msg_parts.extend(process_pdf(fn))
    else:
        with open(fn, mode="rb") as f:
            content = f.read()

        isImage = False
        if isinstance(content, bytes):
            try:
                # try to add as image
                content = encode_image(content)
                isImage = True
            except:
                # not an image, try text
                content = content.decode('utf-8', 'replace')
        else:
            content = str(content)

        if isImage:
            user_msg_parts.append({"type": "input_image",
                                "image_url": content})
        else:
            fn = os.path.basename(fn)
            user_msg_parts.append({"type": "input_text", "text": f"```{fn}\n{content}\n```"})

    return user_msg_parts

def normalize_user_content(content) -> list:
    """Convert chat history entries to OpenAI-style message parts."""
    parts = []

    if hasattr(content, "model_dump"):
        content = content.model_dump()

    if isinstance(content, Component):
        val = getattr(content, "value", None)
        if val is None and hasattr(content, "constructor_args"):
            ca = content.constructor_args
            if isinstance(ca, dict):
                val = ca.get("value")
            elif isinstance(ca, list):
                for entry in ca:
                    if isinstance(entry, dict) and "value" in entry:
                        val = entry["value"]
                        break
        if val is not None:
            content = val

    if isinstance(content, dict):
        if "file" in content and isinstance(content["file"], dict) and content["file"].get("path"):
            parts.extend(encode_file(content["file"]["path"]))
        elif content.get("path"):
            parts.extend(encode_file(content["path"]))
        elif content.get("component"):
            val = content.get("value") or content.get("constructor_args", {}).get("value")
            if isinstance(val, dict) and val.get("path"):
                parts.extend(encode_file(val["path"]))
            else:
                parts.append({"type": "input_text", "text": str(content)})
        else:
            parts.append({"type": "input_text", "text": str(content)})
    elif isinstance(content, Image.Image):
        buf = io.BytesIO()
        fmt = content.format if content.format else "PNG"
        content.save(buf, format=fmt)
        parts.append({"type": "input_image", "image_url": encode_image(buf.getvalue())})
    elif isinstance(content, tuple):
        parts.extend(encode_file(content[0]))
    else:
        parts.append({"type": "input_text", "text": str(content)})

    return parts

def undo(history):
    history.pop()
    return history

def dump(history):
    return str(history)

def load_settings():
    # Dummy Python function, actual loading is done in JS
    pass

def _event_to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"type": getattr(obj, "type", "unknown")}

def save_settings(acc, sec, prompt, temp, tokens, model):  
    # Dummy Python function, actual saving is done in JS  
    pass  

def process_values_js():
    return """
    () => {
        return ["oai_key", "system_prompt"];
    }
    """

def bot(message, history, oai_key, system_prompt, temperature, max_tokens, model, python_use, web_search, *mcp_selected):
    global pending_mcp_request
    try:
        client = OpenAI(
            api_key=oai_key
        )

        approval_items = []
        if pending_mcp_request:
            txt = (message.get("text", "") or "").strip()
            if not txt:
                raise gr.Error("MCP tool call awaiting confirmation. Reply with 'y' to approve or 'n' to deny, optionally followed by a message.")
            flag = txt[0].lower()
            if flag == 'y':
                approve = True
            elif flag == 'n':
                approve = False
            else:
                raise gr.Error("MCP tool call awaiting confirmation. Start your reply with 'y' or 'n'.")
            message["text"] = txt[1:].lstrip()
            approval_items.append(pending_mcp_request)
            approval_items.append({
                "type": "mcp_approval_response",
                "approval_request_id": pending_mcp_request.get("id"),
                "approve": approve,
            })
            pending_mcp_request = None

        if model == "whisper":
            result = ""
            whisper_prompt = system_prompt
            for msg in history:
                role = msg.role if hasattr(msg, "role") else msg["role"]
                content = msg.content if hasattr(msg, "content") else msg["content"]
                if role == "user":
                    if type(content) is tuple:
                        pass
                    else:
                        whisper_prompt += f"\n{content}"
                if role == "assistant":
                        whisper_prompt += f"\n{content}"

            if message["text"]:
                whisper_prompt += message["text"]
            if message.files:
                for file in message.files:
                    audio_fn = os.path.basename(file.path)
                    with open(file.path, "rb") as f:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1", 
                            prompt=whisper_prompt,
                            file=f,
                            response_format="text"
                            )
                    whisper_prompt += f"\n{transcription}"
                    result += f"\n``` transcript {audio_fn}\n {transcription}\n```"
            
            yield gr.ChatMessage(role="assistant", content=result)

        elif model == "gpt-image-1":
            if message.get("files"):
                image_files = []
                for file in message["files"]:
                    image_files.append(open(file, "rb"))

                response = client.images.edit(
                    model=model,
                    image=image_files,
                    prompt=message["text"],
                    quality="high"
                )
                for f in image_files:
                    f.close()
            else:
                response = client.images.generate(
                    model=model,
                    prompt=message["text"],
                    quality="high",
                    moderation="low"
                )
            b64data = response.data[0].b64_json
            img_bytes = base64.b64decode(b64data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            yield gr.ChatMessage(
                role="assistant",
                content=gr.Image(type="pil", value=pil_img)
            )
        else:
            tools = []
            if python_use:
                tools.append({
                    "type": "function",
                    "name": "eval_python",
                    "description": "Evaluate a simple script written in a conservative, restricted subset of Python."
                                "Note: Augmented assignments, in-place operations (e.g., +=, -=), lambdas (e.g. list comprehensions) are not supported. "
                                "Use regular assignments and operations instead. Only 'import math' is allowed. "
                                "Returns: unquoted results without HTML encoding.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "python_source_code": {
                                "type": "string",
                                "description": "The Python script that will run in a RestrictedPython context. "
                                            "Avoid using augmented assignments or in-place operations (+=, -=, etc.), as well as lambdas (e.g. list comprehensions). "
                                            "Use regular assignments and operations instead. Only 'import math' is allowed. Results need to be reported through print()."
                            }
                        },
                        "required": ["python_source_code"]
                    }
                })
            if web_search:
                tools.append({
                    "type": "web_search",
                    "search_context_size": "high"
                    })
            for sel, entry in zip(mcp_selected, mcp_servers):
                if sel:
                    tools.append(to_openai_tool(entry))
            if not tools:
                tools = None

            if log_to_console:
                print(f"bot history: {str(history)}")

            history_openai_format = []
            user_msg_parts = []

            if system_prompt:
                if not model.startswith("o"):
                    role = "system"
                else:
                    role = "developer"

                    if not system_prompt.startswith("Formatting re-enabled"):
                        system_prompt = "Formatting re-enabled\n" + system_prompt
                history_openai_format.append({"role": role, "content": system_prompt})

            for msg in history:
                role = msg.role if hasattr(msg, "role") else msg["role"]
                content = msg.content if hasattr(msg, "content") else msg["content"]

                if role == "user":
                    user_msg_parts.extend(normalize_user_content(content))

                if role == "assistant":
                    if user_msg_parts:
                        history_openai_format.append({"role": "user", "content": user_msg_parts})
                        user_msg_parts = []

                    history_openai_format.append({"role": "assistant", "content": str(content)})

            for item in approval_items:
                history_openai_format.append(item)

            if message["text"]:
                user_msg_parts.append({"type": "input_text", "text": message["text"]})
            if message["files"]:
                for file in message["files"]:
                    user_msg_parts.extend(encode_file(file))
            history_openai_format.append({"role": "user", "content": user_msg_parts})
            user_msg_parts = []

            if log_to_console:
                print(f"br_prompt: {str(history_openai_format)}")

            if model in ["o1", "o1-high", "o1-pro", "o1-2024-12-17", "o3-mini", "o3-mini-high", "o4-mini", "o4-mini-high",
                         "o3", "o3-high"]:
                reasoner = True

                # reasoning effort
                high = False
                if model == "o1-high":
                    model = "o1"
                    high = True
                if model == "o1-pro":
                    model = "o1-pro"
                    high = True
                elif model == "o3-mini-high":
                    model = "o3-mini"
                    high = True
                elif model == "o4-mini-high":
                    model = "o4-mini"
                    high = True
                elif model == "o3-high":
                    model = "o3"
                    high = True
            else:
                reasoner = False

            assistant_msgs = []
            whole_response = ""
            final_msg = None
            mcp_event_msg = None
            loop_tool_calling = True
            while loop_tool_calling:
                request_params = {
                    "model": model,
                    "input": history_openai_format,
                    "store": False
                }
                if reasoner:
                    request_params["reasoning"] = {"effort": "high" if high else "medium"}
                else:
                    request_params["temperature"] = temperature
                if tools:
                    request_params["tools"] = tools
                    request_params["tool_choice"] = "auto"
                if max_tokens > 0:
                    request_params["max_output_tokens"] = max_tokens

                try:
                    stream = client.responses.create(stream=True, **request_params)
                    have_stream = True
                except Exception as e:
                    # fallback to non‑streaming; wrap the single full response in a fake "completed" event
                    # this happens with o3 via un-verified OpenAI accounts
                    response = client.responses.create(stream=False, **request_params)
                    stream = iter([SimpleNamespace(type="response.completed", response=response)])
                    have_stream = False

                loop_tool_calling = False
                for event in stream:
                    if event.type == "response.output_text.delta":
                        if final_msg is None:
                            final_msg = gr.ChatMessage(role="assistant", content="")
                            assistant_msgs.append(final_msg)
                        whole_response += event.delta
                        final_msg.content = whole_response
                        yield assistant_msgs
                    elif event.type in (
                        "response.mcp_list_tools.in_progress",
                        "response.mcp_call.in_progress",
                    ):
                        mcp_event_msg = gr.ChatMessage(
                            role="assistant",
                            content="",
                            metadata={
                                "title": event.type,
                                "id": f"mcp-{getattr(event, 'sequence_number', '')}",
                                "status": "pending",
                            },
                        )
                        assistant_msgs.append(mcp_event_msg)
                        yield assistant_msgs
                    elif event.type in (
                        "response.mcp_list_tools.completed",
                        "response.mcp_list_tools.failed",
                        "response.mcp_call.completed",
                        "response.mcp_call.failed",
                    ):
                        if mcp_event_msg is not None:
                            mcp_event_msg.metadata["status"] = "done"
                        yield assistant_msgs
                    elif event.type == "response.completed":
                        response = event.response
                        outputs = response.output

                        for output in outputs:
                            if output.type == "message":
                                for part in output.content:
                                    if part.type == "output_text":
                                        if not have_stream:
                                            if final_msg is None:
                                                final_msg = gr.ChatMessage(role="assistant", content="")
                                                assistant_msgs.append(final_msg)
                                            whole_response += part.text
                                            final_msg.content = whole_response
                                            yield assistant_msgs

                                        anns = part.annotations
                                        if anns:
                                            link_lines = []
                                            for ann in anns:
                                                if ann.type == "url_citation":
                                                    url = ann.url
                                                    title = ann.title
                                                    link_lines.append(f"- [{title}]({url})")
                                            if link_lines:
                                                link_lines = list(dict.fromkeys(link_lines))
                                                whole_response += "\n\n**Citations:**\n" + "\n".join(link_lines)
                                                if final_msg is None:
                                                    final_msg = gr.ChatMessage(role="assistant", content="")
                                                    assistant_msgs.append(final_msg)
                                                final_msg.content = whole_response
                                                yield assistant_msgs

                            elif output.type == "function_call":
                                if output.name == "eval_python":
                                    try:
                                        history_openai_format.append({
                                            "type": "function_call",
                                            "name": output.name,
                                            "arguments": output.arguments,
                                            "call_id": output.call_id
                                        })

                                        parsed_args = json.loads(output.arguments)
                                        tool_script = parsed_args.get("python_source_code", "")
                                        call_id = output.call_id

                                        parent_msg = gr.ChatMessage(
                                            role="assistant",
                                            content="",
                                            metadata={"title": output.name, "id": call_id, "status": "pending"},
                                        )
                                        assistant_msgs.append(parent_msg)
                                        assistant_msgs.append(
                                            gr.ChatMessage(
                                                role="assistant",
                                                content=f"``` script\n{tool_script}\n```",
                                                metadata={"title": "request", "parent_id": call_id},
                                            )
                                        )
                                        yield assistant_msgs

                                        tool_result = eval_script(tool_script)
                                        result_text = (
                                            tool_result["prints"]
                                            if tool_result["success"]
                                            else tool_result.get("error", "")
                                        )

                                        assistant_msgs.append(
                                            gr.ChatMessage(
                                                role="assistant",
                                                content=f"``` result\n{result_text}\n```",
                                                metadata={"title": "response", "parent_id": call_id, "status": "done"},
                                            )
                                        )
                                        parent_msg.metadata["status"] = "done"
                                        yield assistant_msgs

                                        history_openai_format.append(
                                            {
                                                "type": "function_call_output",
                                                "call_id": output.call_id,
                                                "output": json.dumps(tool_result),
                                            }
                                        )
                                    except Exception as e:
                                        history_openai_format.append({
                                            "type": "function_call_output",
                                            "call_id": output.call_id,
                                            "output": {
                                                    "toolResult": {
                                                        "content": [{"text":  e.args[0]}],
                                                        "status": 'error'
                                                    }
                                            }
                                        })

                                        assistant_msgs.append(
                                            gr.ChatMessage(
                                                role="assistant",
                                                content=f"``` error\n{e.args[0]}\n```",
                                                metadata={"title": "response", "parent_id": call_id, "status": "done"},
                                            )
                                        )
                                        parent_msg.metadata["status"] = "done"
                                        yield assistant_msgs
                                else:
                                        history_openai_format.append(outputs)

                                loop_tool_calling = True
                            elif output.type == "mcp_approval_request":
                                pending_mcp_request = _event_to_dict(output)
                                assistant_msgs.append(
                                    gr.ChatMessage(
                                        role="assistant",
                                        content=(
                                            f"MCP approval needed for {output.name} on {output.server_label} with arguments {output.arguments}."
                                        ),
                                        options=[{"value": "y", "label": "Yes"}, {"value": "n", "label": "No"}],
                                    )
                                )
                                yield assistant_msgs
                                return
                            elif output.type == "mcp_call":
                                history_openai_format.append(_event_to_dict(output))
                                if getattr(output, "output", None) is not None:
                                    assistant_msgs.append(
                                        gr.ChatMessage(
                                            role="assistant",
                                            content=f"``` mcp_result\n{output.output}\n```",
                                            metadata={"title": "response"},
                                        )
                                    )
                                    yield assistant_msgs

                        if log_to_console:
                            print(f"usage: {event.usage}")
                    elif event.type == "response.incomplete":
                        gr.Warning(f"Incomplete response, reason: {event.response.incomplete_details.reason}")
                        if final_msg is None:
                            final_msg = gr.ChatMessage(role="assistant", content="")
                            assistant_msgs.append(final_msg)
                        final_msg.content = whole_response
                        yield assistant_msgs

        if log_to_console:
            print(f"br_result: {str(history)}")

    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def import_history_guarded(oai_key, history, file):
    # check credentials first
    try:
        client = OpenAI(api_key=oai_key)
        client.models.retrieve("gpt-4o")
    except Exception as e:
        raise gr.Error(f"OpenAI login error: {str(e)}")

    # actual import
    chat_history, system_prompt_value = import_history(history, file)

    return chat_history, system_prompt_value, chat_history

with gr.Blocks(delete_cache=(86400, 86400)) as demo:
    gr.Markdown("# OpenAI™️ Chat (Nils' Version™️)")
    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/oai_chat/blob/main/LICENSE).
                    Third party terms and conditions apply, particularly
                    those of the LLM vendor (OpenAI) and hosting provider (Hugging Face). This app and the AI models may make mistakes, so verify any outputs.""")

        oai_key = gr.Textbox(label="OpenAI API Key", elem_id="oai_key", value=load_openai_api_key())
        model = gr.Dropdown(label="Model", value="gpt-4.1", allow_custom_value=True, elem_id="model",
                            choices=["gpt-4o", "gpt-4.1", "gpt-4.5-preview", "o3", "o3-high", "o1-pro", "o1-high", "o1-mini", "o1", "o3-mini-high", "o3-mini", "o4-mini", "o4-mini-high", "chatgpt-4o-latest", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "whisper", "gpt-image-1"])
        system_prompt = gr.TextArea("You are a helpful yet diligent AI assistant. Answer faithfully and factually correct. Respond with 'I do not know' if uncertain.", label="System/Developer Prompt", lines=3, max_lines=250, elem_id="system_prompt")  
        temp = gr.Slider(0, 2, label="Temperature", elem_id="temp", value=1)
        max_tokens = gr.Slider(0, 16384, label="Max. Tokens", elem_id="max_tokens", value=0)
        python_use = gr.Checkbox(label="Python Use", value=False)
        web_search = gr.Checkbox(label="Web Search", value=False)
        mcp_boxes = []
        for entry in mcp_servers:
            label = f"MCP: {entry.get('server_label', entry.get('name'))}"
            mcp_boxes.append(gr.Checkbox(label=label, value=False))
        save_button = gr.Button("Save Settings")
        load_button = gr.Button("Load Settings")  
        dl_settings_button = gr.Button("Download Settings")
        ul_settings_button = gr.Button("Upload Settings")

        load_button.click(load_settings, js="""  
            () => {  
                let elems = ['#oai_key textarea', '#system_prompt textarea', '#temp input', '#max_tokens input', '#model'];
                elems.forEach(elem => {
                    let item = document.querySelector(elem);
                    let event = new InputEvent('input', { bubbles: true });
                    item.value = localStorage.getItem(elem.split(" ")[0].slice(1)) || '';
                    item.dispatchEvent(event);
                });
            }  
        """)

        save_button.click(save_settings, [oai_key, system_prompt, temp, max_tokens, model], js="""  
            (oai, sys, temp, ntok, model) => {  
                localStorage.setItem('oai_key', oai);  
                localStorage.setItem('system_prompt', sys);  
                localStorage.setItem('temp', document.querySelector('#temp input').value);  
                localStorage.setItem('max_tokens', document.querySelector('#max_tokens input').value);  
                localStorage.setItem('model', model);  
            }  
        """) 

        control_ids = [('oai_key', '#oai_key textarea'),
                       ('system_prompt', '#system_prompt textarea'),
                       ('temp', '#temp input'),
                       ('max_tokens', '#max_tokens input'),
                       ('model', '#model')]
        controls = [oai_key, system_prompt, temp, max_tokens, model, python_use, web_search] + mcp_boxes

        dl_settings_button.click(None, controls, js=generate_download_settings_js("oai_chat_settings.bin", control_ids))
        ul_settings_button.click(None, None, None, js=generate_upload_settings_js(control_ids))

    chat = gr.ChatInterface(
        fn=bot,
        multimodal=True,
        additional_inputs=controls,
        autofocus=False,
        type="messages",
        chatbot=gr.Chatbot(elem_id="chatbot", type="messages"),
        textbox=gr.MultimodalTextbox(elem_id="chat_input"),
    )
    chat.textbox.file_count = "multiple"
    chat.textbox.max_plain_text_length = 2**31
    chatbot = chat.chatbot
    chatbot.show_copy_button = True
    chatbot.height = 450


    if dump_controls:
        with gr.Row():
            dmp_btn = gr.Button("Dump")
            txt_dmp = gr.Textbox("Dump")
            dmp_btn.click(dump, inputs=[chatbot], outputs=[txt_dmp])

    with gr.Accordion("Import/Export", open = False):
        import_button = gr.UploadButton("History Import")
        export_button = gr.Button("History Export")
        export_button.click(lambda: None, [chatbot, system_prompt], js=get_export_js())
        dl_button = gr.Button("File download")
        dl_button.click(lambda: None, [chatbot], js="""
            (chat_history) => {
                const languageToExt = {
                    'python': 'py',
                    'javascript': 'js',
                    'typescript': 'ts',
                    'csharp': 'cs',
                    'ruby': 'rb',
                    'shell': 'sh',
                    'bash': 'sh',
                    'markdown': 'md',
                    'yaml': 'yml',
                    'rust': 'rs',
                    'golang': 'go',
                    'kotlin': 'kt'
                };

                const contentRegex = /```(?:([^\\n]+)?\\n)?([\\s\\S]*?)```/;
                const match = contentRegex.exec(chat_history[chat_history.length - 1][1]);
                
                if (match && match[2]) {
                    const specifier = match[1] ? match[1].trim() : '';
                    const content = match[2];
                    
                    let filename = 'download';
                    let fileExtension = 'txt'; // default

                    if (specifier) {
                        if (specifier.includes('.')) {
                            // If specifier contains a dot, treat it as a filename
                            const parts = specifier.split('.');
                            filename = parts[0];
                            fileExtension = parts[1];
                        } else {
                            // Use mapping if exists, otherwise use specifier itself
                            const langLower = specifier.toLowerCase();
                            fileExtension = languageToExt[langLower] || langLower;
                            filename = 'code';
                        }
                    }

                    const blob = new Blob([content], {type: 'text/plain'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${filename}.${fileExtension}`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }
            }
        """)
        import_button.upload(import_history_guarded,
                            inputs=[oai_key, chatbot, import_button],
                            outputs=[chatbot, system_prompt, chat.chatbot_state])

demo.queue(default_concurrency_limit = None).launch()
