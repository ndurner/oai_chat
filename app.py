import gradio as gr
import base64
import os
from pathlib import Path
from openai import OpenAI
import json
from PIL import Image
import io
import asyncio
from settings_mgr import generate_download_settings_js, generate_upload_settings_js
from chat_export import import_history, get_export_js
from mcp_registry import load_registry, get_tools_for_server, call_local_mcp_tool, function_to_mcp_map, shutdown_local_mcp_clients
from gradio.components.base import Component
from types import SimpleNamespace
from dotenv import load_dotenv

from doc2json import process_docx
from code_exec import eval_script

load_dotenv()

dump_controls = False
log_to_console = False

mcp_servers = load_registry()

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

def clear_both_histories():
    """Clear both chatbot display history and OpenAI format history"""
    return [], []

def undo_both_histories(chatbot_history, openai_history):
    """Remove last message from both histories"""

    # remove all Gradio messages until last user message
    while chatbot_history and chatbot_history[-1]["role"] != "user":
        chatbot_history.pop()
    if chatbot_history and chatbot_history[-1]["role"] == "user":
        chatbot_history.pop()

    # remove all messages from OpenAI history until last user message
    while openai_history and not (isinstance(openai_history[-1], dict) and openai_history[-1].get("role") == "user"):
        openai_history.pop()
    if openai_history and isinstance(openai_history[-1], dict) and openai_history[-1].get("role") == "user":
        openai_history.pop()

    return chatbot_history, openai_history

def retry_last_message(chatbot_history, openai_history):
    """Remove last assistant message for retry"""
    if chatbot_history and len(chatbot_history) > 0:
        # Remove the last message if it's from assistant
        last_msg = chatbot_history[-1]
        if hasattr(last_msg, 'role') and last_msg.role == "assistant":
            new_chatbot = chatbot_history[:-1]
            new_openai = openai_history[:-1] if openai_history else []
            return new_chatbot, new_openai
        elif isinstance(last_msg, dict) and last_msg.get('role') == "assistant":
            new_chatbot = chatbot_history[:-1]
            new_openai = openai_history[:-1] if openai_history else []
            return new_chatbot, new_openai
    
    return chatbot_history, openai_history

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

async def bot(message, history, history_openai_format, oai_key, system_prompt, temperature, max_tokens, model, python_use, web_search, *mcp_selected):
    try:
        client = OpenAI(
            api_key=oai_key
        )

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
            approval_items = []
            if history_openai_format:
                last_msg = history_openai_format[-1]
                if last_msg.type == "mcp_approval_request":
                    flag = message[0].lower()
                    if flag == 'y':
                        approve = True
                    elif flag == 'n':
                        approve = False
                    else:
                        raise gr.Error("MCP tool call awaiting confirmation. Start your reply with 'y' or 'n'.")
                    history_openai_format.append({
                        "type": "mcp_approval_response",
                        "approval_request_id": pending_mcp_request.id,
                        "approve": approve,
                    })

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
            # Add selected MCP servers to tools
            for sel, entry in zip(mcp_selected, mcp_servers):
                if sel:
                    tools.extend(await get_tools_for_server(entry))
            if not tools:
                tools = None

            if log_to_console:
                print(f"bot history: {str(history)}")

            instructions = None
            user_msg_parts = []

            if system_prompt:
                    if not system_prompt.startswith("Formatting re-enabled"):
                        instructions = "Formatting re-enabled\n" + system_prompt
                    else:
                        instructions = system_prompt

            # handle chatbot input
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
                         "o3", "o3-high", "o3-low"]:
                reasoner = True

                # reasoning effort
                high = False
                low = False
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
                elif model == "o3-low":
                    model = "o3"
                    low = True
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
                    "store": False,
                    "instructions": instructions
                }
                if reasoner:
                    reasoning_dict = {"summary": "auto"}
                    if high:
                        reasoning_dict["effort"] = "high"
                    elif low:
                        reasoning_dict["effort"] = "low"
                    else:
                        reasoning_dict["effort"] = "medium"
                    request_params["reasoning"] = reasoning_dict
                    request_params["include"] = ["reasoning.encrypted_content"]
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
                        yield assistant_msgs, history_openai_format
                    elif event.type == "response.output_item.added" and event.item.type == "reasoning":
                        summary = ""
                        for str in event.item.summary:
                            if str.type == "summary_text":
                                summary += str.text
                        if summary:
                            rs_msg = gr.ChatMessage(
                                role="assistant",
                                content=summary,
                                metadata={"title": "Reasoning", "id": event.item.id, "status": "done"},
                            )
                            assistant_msgs.append(rs_msg)
                            yield assistant_msgs, history_openai_format
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
                        yield assistant_msgs, history_openai_format
                    elif event.type in (
                        "response.mcp_list_tools.completed",
                        "response.mcp_list_tools.failed",
                        "response.mcp_call.completed",
                        "response.mcp_call.failed",
                    ):
                        if mcp_event_msg is not None:
                            mcp_event_msg.metadata["status"] = "done"
                        yield assistant_msgs, history_openai_format
                    elif event.type == "response.completed":
                        response = event.response
                        outputs = response.output

                        history_openai_format.extend(outputs)

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
                                            yield assistant_msgs, history_openai_format

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
                                                yield assistant_msgs, history_openai_format
                            elif output.type == "function_call":
                                # Check if this is a local MCP tool call
                                function_name = output.name
                                if function_name in function_to_mcp_map:
                                    try:
                                        mcp_info = function_to_mcp_map[function_name]
                                        server_name = mcp_info["server_name"]
                                        tool_name = mcp_info["tool_name"]
                                        
                                        # Find the server entry
                                        server_entry = None
                                        for entry in mcp_servers:
                                            if entry["name"] == server_name:
                                                server_entry = entry
                                                break
                                        
                                        if server_entry:
                                            history_openai_format.append({
                                                "type": "function_call",
                                                "name": function_name,
                                                "arguments": output.arguments,
                                                "call_id": output.call_id
                                            })
                                            
                                            # Parse arguments
                                            arguments = json.loads(output.arguments)
                                            call_id = output.call_id
                                            
                                            # Show the function call to the user
                                            parent_msg = gr.ChatMessage(
                                                role="assistant",
                                                content="",
                                                metadata={"title": f"MCP: {server_name} - {tool_name}", "id": call_id, "status": "pending"},
                                            )
                                            assistant_msgs.append(parent_msg)
                                            assistant_msgs.append(
                                                gr.ChatMessage(
                                                    role="assistant",
                                                    content=f"``` arguments\n{output.arguments}\n```",
                                                    metadata={"title": "request", "parent_id": call_id},
                                                )
                                            )
                                            yield assistant_msgs, history_openai_format
                                            
                                            # Call the MCP tool (async)
                                            try:
                                                tool_result = await call_local_mcp_tool(server_entry, tool_name, arguments)
                                                # Extract text from result
                                                if isinstance(tool_result, list) and tool_result and hasattr(tool_result[0], 'text'):
                                                    result_text = "\n".join([item.text for item in tool_result])
                                                elif hasattr(tool_result, 'text'):
                                                    result_text = tool_result.text
                                                else:
                                                    result_text = str(tool_result)
                                                # Show result to the user
                                                assistant_msgs.append(
                                                    gr.ChatMessage(
                                                        role="assistant",
                                                        content=f"``` result\n{result_text}\n```",
                                                        metadata={"title": "response", "parent_id": call_id, "status": "done"},
                                                    )
                                                )
                                                parent_msg.metadata["status"] = "done"
                                                # Add result to history
                                                history_openai_format.append(
                                                    {
                                                        "type": "function_call_output",
                                                        "call_id": output.call_id,
                                                        "output": result_text,
                                                    }
                                                )
                                                yield assistant_msgs, history_openai_format
                                            except Exception as e:
                                                error_message = str(e)
                                                history_openai_format.append({
                                                    "type": "function_call_output",
                                                    "call_id": output.call_id,
                                                    "output": json.dumps({"error": error_message})
                                                })
                                                assistant_msgs.append(
                                                    gr.ChatMessage(
                                                        role="assistant",
                                                        content=f"``` error\n{error_message}\n```",
                                                        metadata={"title": "response", "parent_id": call_id, "status": "done"},
                                                    )
                                                )
                                                parent_msg.metadata["status"] = "done"
                                                yield assistant_msgs, history_openai_format
                                            
                                            # Need to continue the loop to process the function output
                                            loop_tool_calling = True
                                        else:
                                            # Server entry not found
                                            error_message = f"Server {server_name} not found"
                                            history_openai_format.append({
                                                "type": "function_call_output",
                                                "call_id": output.call_id,
                                                "output": json.dumps({"error": error_message})
                                            })
                                    except Exception as e:
                                        # Some error occurred during processing
                                        error_message = f"Error processing local MCP tool call: {str(e)}"
                                        history_openai_format.append({
                                            "type": "function_call_output",
                                            "call_id": output.call_id,
                                            "output": json.dumps({"error": error_message})
                                        })
                                elif output.name == "eval_python":
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
                                        yield assistant_msgs, history_openai_format

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
                                        yield assistant_msgs, history_openai_format

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
                                        yield assistant_msgs, history_openai_format
                                else:
                                        history_openai_format.append(outputs)

                                loop_tool_calling = True
                            elif output.type == "mcp_approval_request":
                                history_openai_format.append(output)
                                assistant_msgs.append(
                                    gr.ChatMessage(
                                        role="assistant",
                                        content=(
                                            f"MCP approval needed for {output.name} on {output.server_label} with arguments {output.arguments}."
                                        ),
                                        options=[{"value": "y", "label": "Yes"}, {"value": "n", "label": "No"}],
                                    )
                                )
                                yield assistant_msgs, history_openai_format
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
                                    yield assistant_msgs, history_openai_format

                        if log_to_console:
                            print(f"usage: {event.usage}")
                    elif event.type == "response.incomplete":
                        gr.Warning(f"Incomplete response, reason: {event.response.incomplete_details.reason}")
                        if final_msg is None:
                            final_msg = gr.ChatMessage(role="assistant", content="")
                            assistant_msgs.append(final_msg)
                        final_msg.content = whole_response
                        yield assistant_msgs, history_openai_format

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
    chat_history, system_prompt_value, history_openai_format = import_history(history, file)
 
    return chat_history, system_prompt_value, chat_history, history_openai_format

with gr.Blocks(delete_cache=(86400, 86400)) as demo:
    gr.Markdown("# OpenAI™️ Chat (Nils' Version™️)")
    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/oai_chat/blob/main/LICENSE).
                    Third party terms and conditions apply, particularly
                    those of the LLM vendor (OpenAI) and hosting provider (Hugging Face). This app and the AI models may make mistakes, so verify any outputs.""")

        oai_key = gr.Textbox(label="OpenAI API Key", elem_id="oai_key", value=os.environ.get("OPENAI_API_KEY"))
        model = gr.Dropdown(label="Model", value="gpt-4.1", allow_custom_value=True, elem_id="model",
                            choices=["gpt-4o", "gpt-4.1", "gpt-4.5-preview", "o3", "o3-high", "o3-low", "o1-pro", "o1-high", "o1-mini", "o1", "o3-mini-high", "o3-mini", "o4-mini", "o4-mini-high", "chatgpt-4o-latest", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "whisper", "gpt-image-1"])
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

    history_openai_format = gr.State([])
    chat = gr.ChatInterface(
        fn=bot,
        multimodal=True,
        additional_inputs=[history_openai_format] + controls,
        additional_outputs=[history_openai_format],
        autofocus=False,
        type="messages",
        chatbot=gr.Chatbot(elem_id="chatbot", type="messages"),
        textbox=gr.MultimodalTextbox(elem_id="chat_input"),
    )
    chat.textbox.file_count = "multiple"
    chat.textbox.max_plain_text_length = 2**31
    chat.textbox.max_lines = 10
    chatbot = chat.chatbot
    chatbot.show_copy_button = True
    chatbot.height = 450

    # Add event handlers to sync chatbot actions with history_openai_format state
    chatbot.clear(
        fn=clear_both_histories,
        outputs=[chatbot, history_openai_format]
    )
    
    chatbot.undo(
        fn=undo_both_histories,
        inputs=[chatbot, history_openai_format],
        outputs=[chatbot, history_openai_format]
    )
    
    chatbot.retry(
        fn=retry_last_message,
        inputs=[chatbot, history_openai_format],
        outputs=[chatbot, history_openai_format]
    )


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
                            outputs=[chatbot, system_prompt, chat.chatbot_state, history_openai_format])

demo.queue(default_concurrency_limit = None).launch()
