import gradio as gr
import base64
import os
from openai import OpenAI
import json
import fitz
from PIL import Image
import io
from settings_mgr import generate_download_settings_js, generate_upload_settings_js
from chat_export import import_history, get_export_js

from doc2json import process_docx
from code_exec import eval_restricted_script

dump_controls = False
log_to_console = False

temp_files = []

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

def process_pdf_img(pdf_fn: str):
    pdf = fitz.open(pdf_fn)
    message_parts = []

    for page in pdf.pages():
        # Create a transformation matrix for rendering at the calculated scale
        mat = fitz.Matrix(0.6, 0.6)
        
        # Render the page to a pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode image to base64
        base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Construct the data URL
        image_url = f"data:image/png;base64,{base64_encoded}"
        
        # Append the message part
        message_parts.append({
            "type": "text",
            "text": f"Page {page.number} of file '{pdf_fn}'"
        })
        message_parts.append({
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "high"
            }
        })

    pdf.close()

    return message_parts

def encode_file(fn: str) -> list:
    user_msg_parts = []

    if fn.endswith(".docx"):
        user_msg_parts.append({"type": "text", "text": process_docx(fn)})
    elif fn.endswith(".pdf"):
        user_msg_parts.extend(process_pdf_img(fn))
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
            user_msg_parts.append({"type": "image_url",
                                "image_url":{"url": content}})
        else:
            fn = os.path.basename(fn)
            user_msg_parts.append({"type": "text", "text": f"```{fn}\n{content}\n```"})

    return user_msg_parts

def undo(history):
    history.pop()
    return history

def dump(history):
    return str(history)

def load_settings():  
    # Dummy Python function, actual loading is done in JS  
    pass  

def save_settings(acc, sec, prompt, temp, tokens, model):  
    # Dummy Python function, actual saving is done in JS  
    pass  

def process_values_js():
    return """
    () => {
        return ["oai_key", "system_prompt", "seed"];
    }
    """

def bot(message, history, oai_key, system_prompt, seed, temperature, max_tokens, model, python_use):
    try:
        client = OpenAI(
            api_key=oai_key
        )

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

        elif model == "dall-e-3":
            response = client.images.generate(
                model=model,
                prompt=message["text"],
                size="1792x1024",
                quality="hd",
                n=1,
            )
            yield gr.Image(response.data[0].url)
        else:
            seed_i = None
            if seed:
                seed_i = int(seed)

            tools = None if not python_use else [
                {
                    "type": "function",
                    "function": {
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
                    }
                }
            ]

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
                        user_msg_parts.append({"type": "text", "text": content})

                if role == "assistant":
                    if user_msg_parts:
                        history_openai_format.append({"role": "user", "content": user_msg_parts})
                        user_msg_parts = []

                    history_openai_format.append({"role": "assistant", "content": content})

            if message["text"]:
                user_msg_parts.append({"type": "text", "text": message["text"]})
            if message["files"]:
                for file in message["files"]:
                    user_msg_parts.extend(encode_file(file))
            history_openai_format.append({"role": "user", "content": user_msg_parts})
            user_msg_parts = []

            if log_to_console:
                print(f"br_prompt: {str(history_openai_format)}")

            if model in ["o1", "o1-high", "o1-2024-12-17", "o3-mini", "o3-mini-high"]:
                # reasoning effort
                high = False
                if model == "o1-high":
                    model = "o1"
                    high = True
                elif model == "o3-mini-high":
                    model = "o3-mini"
                    high = True

                response = client.chat.completions.create(
                    model=model,
                    messages= history_openai_format,
                    seed=seed_i,
                    reasoning_effort="high" if high else "medium",
                    **({"max_completion_tokens": max_tokens} if max_tokens > 0 else {})
                )

                yield response.choices[0].message.content

                if log_to_console:
                        print(f"usage: {response.usage}")
            else:
                whole_response = ""
                while True:
                    response = client.chat.completions.create(
                        model=model,
                        messages= history_openai_format,
                        temperature=temperature,
                        seed=seed_i,
                        max_tokens=max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                        **{"tools": tools} if python_use else {},
                        tool_choice = "auto" if python_use else None
                    )

                    # Accumulators for partial model responses
                    tool_name_accum = None
                    tool_args_accum = ""
                    tool_call_id = None
                    # process
                    for chunk in response:
                        if chunk.choices:
                            txt = ""
                            for choice in chunk.choices:
                                delta = choice.delta
                                if not delta:
                                    continue

                                cont = delta.content
                                if cont:
                                    txt += cont
                                
                                if delta.tool_calls:
                                    for tc in delta.tool_calls:
                                        if tc.function.name:
                                            tool_name_accum = tc.function.name
                                        if tc.function.arguments:
                                            tool_args_accum += tc.function.arguments
                                        if tc.id:
                                            tool_call_id = tc.id

                            finish_reason = choice.finish_reason
                            if finish_reason:
                                if finish_reason == "tool_calls":
                                    try:
                                        parsed_args = json.loads(tool_args_accum)
                                        tool_script = parsed_args.get("python_source_code", "")

                                        whole_response += f"\n``` script\n{tool_script}\n```\n"
                                        yield whole_response

                                        tool_result = eval_restricted_script(tool_script)

                                        whole_response += f"\n``` result\n{tool_result if not tool_result['success'] else tool_result['prints']}\n```\n"
                                        yield whole_response

                                        history_openai_format.extend([
                                            {
                                                "role": "assistant",
                                                "content": txt,
                                                "tool_calls": [
                                                    {
                                                        "id": tool_call_id,
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_name_accum,
                                                            "arguments": json.dumps(parsed_args)
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                "role": "tool",
                                                "tool_call_id": tool_call_id,
                                                "name": tool_name_accum,
                                                "content": json.dumps(tool_result)
                                            }
                                        ])

                                    except Exception as e:
                                        history_openai_format.extend([{
                                            "role": "tool",
                                            "tool_call_id": tool_call_id,
                                            "name": tool_name_accum,
                                            "content": [
                                                {
                                                    "toolResult": {
                                                        "content": [{"text":  e.args[0]}],
                                                        "status": 'error'
                                                    }
                                                }
                                            ]
                                        }])
                                        whole_response += f"\n``` error\n{e.args[0]}\n```\n"
                                        yield whole_response
                                else:
                                    return
                            else:
                                whole_response += txt
                                yield whole_response
                        if chunk.usage and log_to_console:
                            print(f"usage: {chunk.usage}")

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
    gr.Markdown("# OAI Chat (Nils' Version™️)")
    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/oai_chat/blob/main/LICENSE).
                    Third party terms and conditions apply, particularly
                    those of the LLM vendor (OpenAI) and hosting provider (Hugging Face). This app and the AI models may make mistakes, so verify any outputs.""")

        oai_key = gr.Textbox(label="OpenAI API Key", elem_id="oai_key")
        model = gr.Dropdown(label="Model", value="gpt-4-turbo", allow_custom_value=True, elem_id="model",
                            choices=["gpt-4o", "gpt-4-turbo", "o1-high", "o1-mini", "o1", "o3-mini-high", "o3-mini", "o1-preview", "chatgpt-4o-latest", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4", "gpt-4.5-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106", "whisper", "dall-e-3"])
        system_prompt = gr.TextArea("You are a helpful yet diligent AI assistant. Answer faithfully and factually correct. Respond with 'I do not know' if uncertain.", label="System/Developer Prompt", lines=3, max_lines=250, elem_id="system_prompt")  
        seed = gr.Textbox(label="Seed", elem_id="seed")
        temp = gr.Slider(0, 2, label="Temperature", elem_id="temp", value=1)
        max_tokens = gr.Slider(0, 16384, label="Max. Tokens", elem_id="max_tokens", value=800)
        python_use = gr.Checkbox(label="Python Use", value=False, interactive=False)
        save_button = gr.Button("Save Settings")  
        load_button = gr.Button("Load Settings")  
        dl_settings_button = gr.Button("Download Settings")
        ul_settings_button = gr.Button("Upload Settings")

        load_button.click(load_settings, js="""  
            () => {  
                let elems = ['#oai_key textarea', '#system_prompt textarea', '#seed textarea', '#temp input', '#max_tokens input', '#model'];
                elems.forEach(elem => {
                    let item = document.querySelector(elem);
                    let event = new InputEvent('input', { bubbles: true });
                    item.value = localStorage.getItem(elem.split(" ")[0].slice(1)) || '';
                    item.dispatchEvent(event);
                });
            }  
        """)

        save_button.click(save_settings, [oai_key, system_prompt, seed, temp, max_tokens, model], js="""  
            (oai, sys, seed, temp, ntok, model) => {  
                localStorage.setItem('oai_key', oai);  
                localStorage.setItem('system_prompt', sys);  
                localStorage.setItem('seed', seed);  
                localStorage.setItem('temp', document.querySelector('#temp input').value);  
                localStorage.setItem('max_tokens', document.querySelector('#max_tokens input').value);  
                localStorage.setItem('model', model);  
            }  
        """) 

        control_ids = [('oai_key', '#oai_key textarea'),
                       ('system_prompt', '#system_prompt textarea'),
                       ('seed', '#seed textarea'),
                       ('temp', '#temp input'),
                       ('max_tokens', '#max_tokens input'),
                       ('model', '#model')]
        controls = [oai_key, system_prompt, seed, temp, max_tokens, model, python_use]

        dl_settings_button.click(None, controls, js=generate_download_settings_js("oai_chat_settings.bin", control_ids))
        ul_settings_button.click(None, None, None, js=generate_upload_settings_js(control_ids))

    chat = gr.ChatInterface(fn=bot, multimodal=True, additional_inputs=controls, autofocus = False, type = "messages")
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
                            outputs=[chatbot, system_prompt])

demo.unload(lambda: [os.remove(file) for file in temp_files])
demo.queue(default_concurrency_limit = None).launch()