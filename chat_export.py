import json
import base64
import os, io
import mimetypes
from PIL import Image
import gradio as gr

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
                            # Create gr.Image from data URI
                            image_data = base64.b64decode(item['image_url']['url'].split(',')[1])
                            img = Image.open(io.BytesIO(image_data))
                            chat_history.append({
                                "role": msg['role'],
                                "content": gr.Image(value=img)
                            })
                        elif item.get('type', '') == 'file':
                            # Handle file content with gr.File
                            fname = os.path.basename(item['file'].get('name', f'download{msg_num}'))
                            dir_path = os.path.dirname(file.name)
                            temp_path = os.path.join(dir_path, fname)
                            file_data = base64.b64decode(item['file']['url'].split(',')[1])
                            if (len(file_data) > 15e6):
                                raise ValueError(f"file content `{fname}` larger than 15 MB")
                            
                            with open(temp_path, "wb") as tempf:
                                tempf.write(file_data)
                            chat_history.append({
                                "role": msg['role'],
                                "content": gr.File(value=temp_path, 
                                                 label=fname)
                            })
                        else:
                            chat_history.append(msg)
                else:
                    chat_history.append(msg)
                    
            elif msg['role'] == 'assistant':
                chat_history.append(msg)

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
        # Convert tuple/pair format to messages format
        for pair in legacy_history:
            if pair[0]:  # User message
                if isinstance(pair[0], dict) and 'file' in pair[0]:
                    if 'data' in pair[0]['file']:
                        # Legacy format with embedded data
                        file_data = pair[0]['file']['data']
                        mime_type = file_data.split(';')[0].split(':')[1]
                        
                        if mime_type.startswith('image/'):
                            image_data = base64.b64decode(file_data.split(',')[1])
                            img = Image.open(io.BytesIO(image_data))
                            chat_history.append({
                                "role": "user",
                                "content": gr.Image(value=img)
                            })
                        else:
                            fname = pair[0]['file'].get('name', 'download')
                            dir_path = os.path.dirname(file.name)
                            temp_path = os.path.join(dir_path, fname)
                            file_data = base64.b64decode(file_data.split(',')[1])
                            
                            with open(temp_path, "wb") as tempf:
                                tempf.write(file_data)
                            chat_history.append({
                                "role": "user",
                                "content": gr.File(value=temp_path, 
                                                 label=fname)
                            })
                    else:
                        # Keep as-is but convert to message format
                        chat_history.append({
                            "role": "user",
                            "content": pair[0]
                        })
                else:
                    chat_history.append({
                        "role": "user",
                        "content": pair[0]
                    })
            
            if pair[1]:  # Assistant message
                chat_history.append({
                    "role": "assistant",
                    "content": pair[1]
                })

    return chat_history, system_prompt_value

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