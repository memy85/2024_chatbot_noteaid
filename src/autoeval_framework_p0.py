
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import pandas as pd
import json
import pickle
import argparse 
from utils import *

import time
import openai
from openai import RateLimitError
from openai import OpenAI

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
filename = os.path.basename(__file__)
logging = config.load_logger(filename)
prompt = config.load_prompts("generate_synthetic_notes.txt")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def query_chatgpt(prompt) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : prompt}
            ],
            max_tokens=800,
        )
    except :
        logging("API Call did not work..")

    return completion.choices[0].message.content

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_flag", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--number_of_notes", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__" :

    args = parse_args()
    # data_type = args.type
    # logging.debug(f"arguments are data_type : {data_type}")
    openai_flag = args.openai_flag
    counts = args.number_of_notes
    logging.debug(f"arguments are open_ai : {openai_flag}")

    # MODEL_PATH = config.model_path("llama3.2-3B")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if openai_flag : 
        # we use chatgpt-4o-mini
        pass

    else :
        model = pipeline("text-generation", model=MODEL_PATH, device_map="auto")
    
    synthetic_notes = []
    for i in range(counts) : 
        case = {}
        id = f"SYNNOTE{i}"
        case["ID"] = id
        if openai_flag :
            out = query_chatgpt(prompt)
            time.sleep(2)
        else :
            out = model(prompt.format(conversation), return_full_text=False, max_new_tokens=6)
        case["note"] = out
        synthetic_notes.append(case)

    if openai_flag : 
        with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_notes_{counts}.pkl"), 'wb') as f :
            pickle.dump(synthetic_notes, f)
    else :  
        with open(DATA_PATH.joinpath(f"llama_generated_synthetic_notes_{counts}.pkl"), 'wb') as f :
            pickle.dump(synthetic_notes, f)

