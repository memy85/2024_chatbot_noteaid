 
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
prompt = config.load_prompts("generate_chats.txt")

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

def query_chatgpt_to_generate_chat(prompt) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : prompt}
            ],
            # max_tokens=300,
        )
    except :
        logging("API Call did not work..")
        return -1

    return completion.choices[0].message.content

def query_chatgpt_to_generate_qa(prompt) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : prompt}
            ],
            # max_tokens=300,
        )
    except :
        logging("API Call did not work..")
        return -1

    return completion.choices[0].message.content

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_chats", default=2, type=int)
    return parser.parse_args()

if __name__ == "__main__" :

    args = parse_args()
    number_of_chats = args.number_of_chats

    path = DATA_PATH.joinpath("gpt4o_mini_generated_synthetic_notes_10000.pkl")
    with open(path, 'rb') as f :
        synthetic_notes = pickle.load(f)

    path = DATA_PATH.joinpath("synthetic_qa_10000.pkl")
    with open(path, 'rb') as f :
        synthetic_qa = pickle.load(f)
    
    conversations = []
    count = 0
    for i, note in enumerate(synthetic_notes): 
        p = prompt.format(discharge_note=note['note'], qa=str(synthetic_qa[i]))
        out = query_chatgpt(p)
        time.sleep(2)

        if out == -1 :
            break

        conversations.append({"messages" : out})
        count += 1

        if count == number_of_chats :
            break

    with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_conversation_{count}.pkl"), 'wb') as f :
        pickle.dump(conversations, f)


