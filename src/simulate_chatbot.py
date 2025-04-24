import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
import torch
import argparse
from transformers import AutoModelForCausalLM, pipeline, GenerationConfig, TextStreamer
from peft.peft_model import PeftModelForCausalLM, PeftModel
from peft import PeftConfig, AutoPeftModelForCausalLM

from utils import *

warnings.filterwarnings("ignore")

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

def load_model(MODEL_PATH) :
    # model_path = config.model_path('llama3.2-1B')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # model = AutoPeftModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
    return model, tokenizer

def arguments() :
    parser = argparse.ArgumentParser()
    # parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__" :

    args = arguments()
    model_path = args.model_path

    # test_path = PROJECT_PATH.joinpath("model/gguf/llama3.2-3B/chatbot11")
    # model, tokenizer = load_model(test_path)

    model, tokenizer = load_model(model_path)
    # model = model.bfloat16()
    # streamer = TextStreamer(tokenizer)
    with open(PROJECT_PATH.joinpath("discharge_notes/case1.txt"), "r") as f :
        discharge_note = f.read()

    messages = [
            {"role":"system","content": f"You are a helpful medical educator agent. You will generate short and easy to understand chat. Here is the discharge not of the patient. \nDischarge Note : \n{discharge_note} \n\nStart the conversation saying 'hello how are you'." },
    ]
    turn = 0

    generation_config = {
        "max_length" : 16000,
        "max_new_tokens" : 200,
        "repetition_penalty" : 1.8,
        "top_k" : 50,
        "temperature" : 0.7,
        "early_stopping" : False,
        "length_penalty" : -1,
        "num_beams" : 10,
        # "stop_strings" : ["<end of conversation>"]
        # "eos_token_id" : tokenizer.eos_token_id
    }

    while True :
        # Start from chatbot
        # print(f"This is {turn}th turn")
        
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()

        output_ids = model.generate(tokenized_chat, **generation_config) # tokenizer=tokenizer) # streamer=streamer
        arr_output = output_ids.detach().cpu().numpy()
        start_of_generate_index = tokenized_chat.shape[1]
        response = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]

        print(f"\nassistant: {response} \n")
        messages.append({"role":"assistant", "content":response})

        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"] :
            print("Exiting conversation...")
            break
        
        # if "<EOC>" in user_input :
        #     messages.append({"role":"user","content":user_input})

        messages.append({"role":"user","content":user_input})
        turn += 1
    torch.cuda.empty_cache()
