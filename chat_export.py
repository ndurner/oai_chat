import json
import base64
import os, io
import mimetypes
from PIL import Image

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
                            # Save image to a temporary file
                            data_uri = item['image_url']['url']
                            img_bytes = base64.b64decode(data_uri.split(',')[1])
                            mime_type = data_uri.split(';')[0].split(':')[1]
                            ext = mimetypes.guess_extension(mime_type) or '.png'
                            fname = f"download{msg_num}{ext}"
                            dir_path = os.path.dirname(file.name)
                            temp_path = os.path.join(dir_path, fname)
                            with open(temp_path, 'wb') as tempf:
                                tempf.write(img_bytes)
                            chat_history.append({
                                "role": msg['role'],
                                "content": {"path": temp_path}
                            })
                        elif item.get('type', '') == 'file':
                            # Handle file content
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
                                "content": {"path": temp_path}
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
                            image_bytes = base64.b64decode(file_data.split(',')[1])
                            ext = mimetypes.guess_extension(mime_type) or '.png'
                            fname = f"download{msg_num}{ext}"
                        else:
                            fname = pair[0]['file'].get('name', 'download')
                            image_bytes = base64.b64decode(file_data.split(',')[1])

                        dir_path = os.path.dirname(file.name)
                        temp_path = os.path.join(dir_path, fname)
                        with open(temp_path, 'wb') as tempf:
                            tempf.write(image_bytes)
                        chat_history.append({
                            "role": "user",
                            "content": {"path": temp_path}
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