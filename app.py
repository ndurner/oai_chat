import gradio as gr
import base64
import os
from openai import OpenAI
import json
from PIL import Image
import io
from settings_mgr import generate_download_settings_js, generate_upload_settings_js
from chat_export import import_history, get_export_js
from mcp_registry import load_registry, to_openai_tool
from types import SimpleNamespace

from doc2json import process_docx
from code_exec import eval_restricted_script

dump_controls = False
log_to_console = False

temp_files = []
mcp_servers = load_registry()
pending_mcp_request = None

approval_modal_js = """
() => {
    window.oai_mcp_modal = {
        show: (msg) => {
            const modal = document.getElementById('mcp_modal');
            if (!modal) return;
            document.getElementById('mcp_modal_text').innerText = msg;
            modal.style.display = 'flex';
        },
        hide: () => {
            const modal = document.getElementById('mcp_modal');
            if (!modal) return;
            modal.style.display = 'none';
            document.getElementById('mcp_modal_input').value = '';
        },
        approve: () => {
            const txt = document.getElementById('mcp_modal_input').value;
            const tb = document.querySelector('#chat_input textarea');
            tb.value = 'y ' + txt;
            tb.dispatchEvent(new Event('input', {bubbles: true}));
            window.oai_mcp_modal.hide();
            document.querySelector('#chat_input button').click();
        },
        deny: () => {
            const txt = document.getElementById('mcp_modal_input').value;
            const tb = document.querySelector('#chat_input textarea');
            tb.value = 'n ' + txt;
            tb.dispatchEvent(new Event('input', {bubbles: true}));
            window.oai_mcp_modal.hide();
            document.querySelector('#chat_input button').click();
        }
    };
    document.getElementById('mcp_modal_approve').onclick = window.oai_mcp_modal.approve;
    document.getElementById('mcp_modal_deny').onclick = window.oai_mcp_modal.deny;
}
"""

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
                raise gr.Error("MCP tool call awaiting confirmation. Use the dialog to approve or deny, optionally adding a message.")
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
                content = msg["content"]
                if msg["role"] == "user":
                    if type(content) is tuple:
                        pass
                    else:
                        whisper_prompt += f"\n{content}"
                if msg["role"] == "assistant":
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
            
            yield result

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
                role = msg["role"]
                content = msg["content"]

                if role == "user":
                    if isinstance(content, gr.File) or isinstance(content, gr.Image):
                        user_msg_parts.extend(encode_file(content.value['path']))
                    elif isinstance(content, tuple):
                        user_msg_parts.extend(encode_file(content[0]))
                    else:
                        user_msg_parts.append({"type": "input_text", "text": content})

                if role == "assistant":
                    if user_msg_parts:
                        history_openai_format.append({"role": "user", "content": user_msg_parts})
                        user_msg_parts = []

                    history_openai_format.append({"role": "assistant", "content": content})

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

            whole_response = ""
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
                        whole_response += event.delta
                        yield whole_response
                    elif event.type == "response.completed":
                        response = event.response
                        outputs = response.output

                        for output in outputs:
                            if output.type == "message":
                                for part in output.content:
                                    if part.type == "output_text":
                                        if not have_stream:
                                            # response text was not collected through streaming events, so get it here
                                            whole_response += part.text
                                            yield whole_response

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
                                                yield whole_response

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

                                        whole_response += f"\n``` script\n{tool_script}\n```\n"
                                        yield whole_response

                                        tool_result = eval_restricted_script(tool_script)

                                        whole_response += f"\n``` result\n{tool_result if not tool_result['success'] else tool_result['prints']}\n```\n"
                                        yield whole_response

                                        history_openai_format.append({
                                            "type": "function_call_output",
                                            "call_id": output.call_id,
                                            "output": json.dumps(tool_result)
                                        })
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

                                        whole_response += f"\n``` error\n{e.args[0]}\n```\n"
                                        yield whole_response
                                else:
                                        history_openai_format.append(outputs)

                                loop_tool_calling = True
                            elif output.type == "mcp_approval_request":
                                pending_mcp_request = _event_to_dict(output)
                                whole_response += (f"\nMCP approval needed for {output.name}"
                                                 f" on {output.server_label} with arguments {output.arguments}.")
                                yield whole_response
                                return
                            elif output.type == "mcp_call":
                                history_openai_format.append(_event_to_dict(output))
                                if getattr(output, "output", None) is not None:
                                    whole_response += f"\n``` mcp_result\n{output.output}\n```\n"
                                    yield whole_response
                                loop_tool_calling = True
                        
                        if log_to_console:
                            print(f"usage: {event.usage}")
                    elif event.type == "response.incomplete":
                        gr.Warning(f"Incomplete response, reason: {event.response.incomplete_details.reason}")
                        yield whole_response

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
    return import_history(history, file)

with gr.Blocks(delete_cache=(86400, 86400)) as demo:
    gr.Markdown("# OpenAI™️ Chat (Nils' Version™️)")
    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/oai_chat/blob/main/LICENSE).
                    Third party terms and conditions apply, particularly
                    those of the LLM vendor (OpenAI) and hosting provider (Hugging Face). This app and the AI models may make mistakes, so verify any outputs.""")

        oai_key = gr.Textbox(label="OpenAI API Key", elem_id="oai_key")
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

    modal = gr.HTML("""
        <div id='mcp_modal' style='display:none; position:fixed; z-index:1000; top:0; left:0; width:100%; height:100%;
             background:rgba(0,0,0,0.5); justify-content:center; align-items:center;'>
          <div style='background:white; padding:20px; border-radius:8px; max-width:90%;'>
            <div id='mcp_modal_text' style='white-space:pre-wrap;'></div>
            <textarea id='mcp_modal_input' rows='2' style='width:100%; margin-top:10px;' placeholder='Optional reply'></textarea>
            <div style='text-align:right; margin-top:10px;'>
              <button id='mcp_modal_deny'>Deny</button>
              <button id='mcp_modal_approve'>Approve</button>
            </div>
          </div>
        </div>
        """)

    chat = gr.ChatInterface(fn=bot, multimodal=True, additional_inputs=controls, autofocus=False, type="messages",
                            chatbot=gr.Chatbot(elem_id="chatbot", type="messages"),
                            textbox=gr.MultimodalTextbox(elem_id="chat_input"),
                            js=approval_modal_js)
    chat.textbox.file_count = "multiple"
    chat.textbox.max_plain_text_length = 2**31
    chatbot = chat.chatbot
    chatbot.show_copy_button = True
    chatbot.height = 450
    chatbot.change(js="""
        (hist) => {
            if (hist && hist.length > 0) {
                let last = hist[hist.length - 1];
                if (last.role === 'assistant' && typeof last.content === 'string' && last.content.includes('MCP approval needed for')) {
                    window.oai_mcp_modal.show(last.content);
                }
            }
        }
    """)

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
                            outputs=[chatbot, system_prompt])

demo.unload(lambda: [os.remove(file) for file in temp_files])
demo.queue(default_concurrency_limit = None).launch()