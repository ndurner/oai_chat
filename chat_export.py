import json
import base64
import os, io
from PIL import Image
import gradio as gr
from gradio import processing_utils, utils
import openai

def import_history(history, file):
    if os.path.getsize(file.name) > 100e6:
        raise ValueError("History larger than 100 MB")

    with open(file.name, mode="rb") as f:
        content = f.read().decode('utf-8', 'replace')

    import_data = json.loads(content)
    
    # Handle different import formats
    if 'messages' in import_data:
        # New OpenAI-style format
        messages = import_data['messages']
        system_prompt_value = ''
        chat_history = []
        openai_history = []
        
        msg_num = 1
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt_value = msg['content']
                continue
                
            if msg['role'] == 'user':
                content = msg['content']
                if isinstance(content, list):
                    for item in content:
                        if item.get('type', '') == 'image_url':
                            data_uri = item['image_url']['url']
                            img_bytes = base64.b64decode(data_uri.split(',')[1])
                            fname = f"img{msg_num}.webp"
                            cache_path = processing_utils.save_bytes_to_cache(img_bytes, fname, utils.get_upload_folder())
                            chat_history.append({
                                "role": msg['role'],
                                "content": {"path": cache_path}
                            })
                            openai_history.append({
                                "role": "user",
                                "content": [{
                                    "type": "input_image",
                                    "detail": "high",
                                    "image_url": data_uri
                                }]
                            })
                        elif item.get('type', '') == 'file':
                            fname = os.path.basename(item['file'].get('name', f'download{msg_num}'))
                            file_data = base64.b64decode(item['file']['url'].split(',')[1])
                            if (len(file_data) > 15e6):
                                raise ValueError(f"file content `{fname}` larger than 15 MB")

                            cache_path = processing_utils.save_bytes_to_cache(file_data, fname, utils.get_upload_folder())
                            chat_history.append({
                                "role": msg['role'],
                                "content": {"path": cache_path}
                            })
                            openai_history.append({
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "content": file_data
                                }]
                            })
                        else:
                            # untested - does not happen?
                            chat_history.append(item["content"])
                            openai_history.append({
                                "role": "user",
                                "content": item["content"]
                            })
                else:
                    chat_history.append(msg)
                    openai_history.append({
                        "role": "user",
                        "content": content
                    })
                    
            elif msg['role'] == 'assistant':
                chat_history.append(msg)
                openai_history.append(openai.types.responses.ResponseOutputMessage(
                    id=f"msg_{msg_num}",
                    role="assistant",
                    content=[openai.types.responses.ResponseOutputText(type="output_text", text=msg['content'], annotations=[])],
                    status="completed",
                    type="message"
                ))

            msg_num = msg_num + 1
            
    else:
        # Legacy format handling
        if 'history' in import_data:
            legacy_history = import_data['history']
            system_prompt_value = import_data.get('system_prompt', '')
        else:
            legacy_history = import_data
            system_prompt_value = ''
        
        chat_history = []
        openai_history = []
        msg_num = 1
        # Convert tuple/pair format to messages format
        for pair in legacy_history:
            if pair[0]:  # User message
                if isinstance(pair[0], dict) and 'file' in pair[0]:
                    if 'data' in pair[0]['file']:
                        # Legacy format with embedded data
                        file_data = pair[0]['file']['data']
                        mime_type = file_data.split(';')[0].split(':')[1]
                        
                        if mime_type.startswith('image/'):
                            image_bytes = base64.b64decode(file_data.split(',')[1])
                            fname = 'legacy_img.webp'
                            cache_path = processing_utils.save_bytes_to_cache(image_bytes, fname, utils.get_upload_folder())
                            chat_history.append({
                                "role": "user",
                                "content": {"path": cache_path}
                            })
                            openai_history.append({
                                "role": "user",
                                "content": [{
                                    "type": "input_image",
                                    "detail": "high",
                                    "image_url": file_data
                                }]
                            })
                        else:
                            fname = pair[0]['file'].get('name', 'download')
                            file_bytes = base64.b64decode(file_data.split(',')[1])
                            cache_path = processing_utils.save_bytes_to_cache(file_bytes, fname, utils.get_upload_folder())
                            chat_history.append({
                                "role": "user",
                                "content": {"path": cache_path}
                            })
                            openai_history.append({
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "content": file_bytes
                                }]
                            })
                    else:
                        # Keep as-is but convert to message format
                        chat_history.append({
                            "role": "user",
                            "content": pair[0]
                        })
                        openai_history.append({
                            "role": "user",
                            "content": pair[0]
                        })
                else:
                    chat_history.append({
                        "role": "user",
                        "content": pair[0]
                    })
                    openai_history.append({
                        "role": "user",
                        "content": pair[0]
                    })
            
            if pair[1]:  # Assistant message
                chat_history.append({
                    "role": "assistant",
                    "content": pair[1]
                })
                openai_history.append(openai.types.responses.ResponseOutputMessage(
                    id=f"msg_{msg_num}",
                    role="assistant",
                    content=[openai.types.responses.ResponseOutputText(type="output_text", text=pair[1], annotations=[])],
                    status="completed",
                    type="message"
                ))
            
            msg_num = msg_num + 1

    return chat_history, system_prompt_value, openai_history

def get_export_js():
    return """
    async (chat_history, system_prompt) => {
        let messages = [];
        
        if (system_prompt) {
            messages.push({
                "role": "system",
                "content": system_prompt
            });
        }

        async function processFile(file_url) {
            const response = await fetch(file_url);
            const blob = await response.blob();
            return new Promise((resolve) => {
                const reader = new FileReader();
                reader.onloadend = () => resolve({
                    data: reader.result,
                    type: blob.type
                });
                reader.onerror = (error) => resolve(null);
                reader.readAsDataURL(blob);
            });
        }

        for (let message of chat_history) {
            if (!message.role || !message.content) continue;

            if (message.content && typeof message.content === 'object') {
                if (message.content.file) {
                    try {
                        const file_data = await processFile(message.content.file.url);
                        if (!file_data) continue;

                        if (file_data.type.startsWith('image/')) {
                            messages.push({
                                "role": message.role,
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {
                                        "url": file_data.data
                                    }
                                }]
                            });
                        } else {
                            const fileLink = document.querySelector(`a[data-testid="chatbot-file"][download][href*="${message.content.file.url.split('/').pop()}"]`);
                            const fileName = fileLink ? fileLink.getAttribute('download') : (message.content.file.name || "download");
                            
                            messages.push({
                                "role": message.role,
                                "content": [{
                                    "type": "file",
                                    "file": {
                                        "url": file_data.data,
                                        "name": fileName,
                                        "mime_type": file_data.type
                                    }
                                }]
                            });
                        }
                    } catch (error) {}
                }
            } else {
                messages.push({
                    "role": message.role,
                    "content": message.content
                });
            }
        }

        const export_data = { messages };
        const blob = new Blob([JSON.stringify(export_data)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_history.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    """