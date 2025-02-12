
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os, sys
os.sys.path.append("/home/wjang/2024_chatbot_noteaid/src")

from src.utils import *

config = load_config()
PROJECT_PATH = config.project
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MODEL_PATH = PROJECT_PATH.joinpath("model/")

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests (for frontend integration)

# Load the fine-tuned LLaMA model
model_name = MODEL_PATH.joinpath("llama3.2-1B/chatbot4/checkpoint-3330")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Default system prompt
system_prompt = "You are an AI assistant for physicians. Please analyze patient discharge summaries.\n"

@app.route("/upload", methods=["POST"])
def upload_discharge_note():
    global system_prompt
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    content = file.read().decode("utf-8")
    system_prompt = f"Patient Discharge Note:\n{content}\n\nYou are an AI assistant for physicians.\n"
    return jsonify({"message": "File uploaded and system prompt updated"})

@app.route("/chat", methods=["POST"])
def chat():
    global system_prompt
    data = request.json
    query = data.get("query", "")
    history = data.get("history", [])

    prompt = system_prompt + "\n" + "".join([f"User: {msg[0]}\nAssistant: {msg[1]}\n" for msg in history]) + f"User: {query}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=1024, do_sample=True, top_p=0.9, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

    history.append((query, response))
    return jsonify({"response": response, "history": history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

