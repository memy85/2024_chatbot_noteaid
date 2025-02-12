import gradio as gr
import requests
import json

def upload_file(file):
    """Uploads discharge note and updates the system prompt."""
    files = {"file": (file.name, file.read(), "text/plain")}
    response = requests.post("http://localhost:8000/upload", files=files)
    return response.json()["message"]

def chat_fn(query, history):
    """Sends user messages to Flask API and retrieves responses."""
    data = {"query": query, "history": json.dumps(history)}
    response = requests.post("http://localhost:8000/chat", json=data)
    return response.json()["response"], response.json()["history"]

# Define the Gradio UI
with gr.Blocks() as chat_ui:
    gr.Markdown("# 🏥 Physician AI Chatbot")
    
    # File Upload Section
    with gr.Row():
        file_uploader = gr.File(label="Upload Patient Discharge Note (.txt)", type="file")
        file_status = gr.Textbox(label="File Upload Status", interactive=False)
        file_uploader.upload(upload_file, file_uploader, file_status)

    # Chatbot Interface
    chatbot = gr.ChatInterface(chat_fn)

# Launch the Gradio UI
chat_ui.launch(server_name="0.0.0.0", server_port=7860)
