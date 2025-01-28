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
import backoff
from openai import OpenAI


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
filename = os.path.basename(__file__)
logging = config.load_logger(filename)
prompt = config.load_prompts("criteria_classification.txt")

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

def query_chatgpt(prompt, sentence) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : prompt.format(sentence)}
            ],
            max_tokens=6,
        )
    except :
        logging("API Call did not work..")

    return completion.choices[0].message.content

def load_datasets() :

    data = load_dataset("json", data_files={"train" : DATA_PATH.joinpath("train_conversation.jsonl").as_posix(),
                                    "test" : DATA_PATH.joinpath("test_conversation.jsonl").as_posix()})
    return data

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='train')
    parser.add_argument("--openai_flag", type=str2bool, nargs='?', const=True, default=False)
    return parser.parse_args()

if __name__ == "__main__" :

    args = parse_args()
    data_type = args.type
    openai_flag = args.openai_flag
    logging.debug(f"arguments are data_type : {data_type}")
    logging.debug(f"arguments are open_ai : {openai_flag}")

    if openai_flag : 
        pass

    else :
        MODEL_PATH = config.model_path("llama3.2-3B")
        model = pipeline("text-generation", model=MODEL_PATH, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    dataset = load_datasets()
    # dataset = data.map(lambda x : {"formatted_chat": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})

    categories_for_dataset = []
    for sample in dataset[data_type]['messages'] :
        outs = []
        for s in sample :
            if s['role'] == "system" :
                continue
            if openai_flag :
                out = query_chatgpt(prompt, s['content'])
                outs.append(out)
                time.sleep(2)
            else :
                text = s['content']
                out = model(prompt.format(text), return_full_text=False, max_new_tokens=6)
                outs.append(out[0]['generated_text'])

        categories_for_dataset.append(outs)
    if openai_flag : 
        with open(DATA_PATH.joinpath(f"evaluation_results_gpt4o_{data_type}.pkl"), 'wb') as f :
            pickle.dump(categories_for_dataset, f)
    else :  
        with open(DATA_PATH.joinpath(f"evaluation_results_{data_type}.pkl"), 'wb') as f :
            pickle.dump(categories_for_dataset, f)
