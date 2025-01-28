import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
import torch
from transformers import AutoModelForCausalLM, pipeline, GenerationConfig
from peft.peft_model import PeftModelForCausalLM, PeftModel
from peft import PeftConfig, AutoPeftModelForCausalLM

from utils import *

warnings.filterwarnings("ignore")

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

def load_model(MODEL_PATH) :
    model_path = config.model_path('llama3.2-1B')
    model = AutoPeftModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

if __name__ == "__main__" :

    test_path = PROJECT_PATH.joinpath("model/llama3.2-1B/chatbot4/checkpoint-3330")
    model, tokenizer = load_model(test_path)
    messages = [
        {"role":"system","content": "You are a helpful nursing assistant. You will generate short and easy to understand chat."},
    ]
    turn = 0

    generation_config = {
        "max_length" : 60000,
        "max_new_tokens" : 100,
        "repetition_penalty" : 1.2,
        "top_k" : 50,
        "temperature" : 0.9,
        "early_stopping" : True,
        "num_beams" : 5,
        "eos_token_id" : tokenizer.eos_token_id
    }

    while True :
        # Start from chatbot
        # print(f"This is {turn}th turn")
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()

        output_ids = model.generate(tokenized_chat, **generation_config) 
        arr_output = output_ids.detach().cpu().numpy()
        start_of_generate_index = tokenized_chat.shape[1]
        response = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]

        print(f"assistant: {response} \n")
        messages.append({"role":"assistant", "content":response})

        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"] :
            print("Exiting conversation...")
            break

        messages.append({"role":"user","content":user_input})
        turn += 1
    torch.cuda.empty_cache()
