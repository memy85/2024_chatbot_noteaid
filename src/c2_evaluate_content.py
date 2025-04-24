
import os,sys
from datasets import load_dataset, Dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import json
import argparse
import textstat
import time
import openai
from openai import OpenAI

from utils import *

config = load_config()
filename = os.path.basename(__file__)
logging = config.load_logger(filename)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
# MAX_LENGTH = 60000
# checkpoint = 3330

models = [
    # "gpt-4o-mini", 
    # "llama3.2-3B", 
    # "llama3.2-3B-lora", 
    "llama3.2-3B-sft", 
    # "llama3.2-3B-lora-ppo", 
    "llama3.2-3B-lora-grpo"
    ]

CONTENT_CLASSIFICATION_PROMPT = config.load_prompts("criteria_classification.txt")

def reformat_conversation(history) :
    conversation_history = []
    for turn in history :
        if turn['role'] == 'assistant' :
            conversation_history.append(f"{turn['content']}")
    return conversation_history


def load_conversation_history(model, dataset) :
    # conversation_history = []
    # for model in models.keys() : 
    with open(DATA_PATH.joinpath(f"evaluation_conversation_history_{model}_{dataset}.pkl"), 'rb') as f :
        data = pickle.load(f)

        reformatted_data =  []
        for history in data : 
            out = reformat_conversation(history['conversation'])
            reformatted_data.append(out)

        # conversation_history[model] = reformatted_data

    return reformatted_data 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def query_chatgpt(client, message) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = message,
            max_tokens=6,
        )
    except :
        # logging.debug("API Call did not work..")
        print("API Call did not work")

    return completion.choices[0].message.content


def arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=False)
    # parser.add_argument("--model", type=str, default='llama3.2-3B')
    parser.add_argument("--dataset", type=str, default='gold')
    args = parser.parse_args()
    return args


def main() :
    args = arguments()
    TEST_FLAG = args.test
    # model = args.model
    dataset = args.dataset
    # test_data = form_dataset()
    # model_name = 'llama3.2-3B-lora'
    # dataset = 'gold'

    for model_name in models :
        
        histories = load_conversation_history(model_name, dataset)
        model_classification = []
        for i, history in enumerate(histories) :
            for turn in history : 
                # turn = histories[0][1]
                message = [{"role" : "system", "content" : CONTENT_CLASSIFICATION_PROMPT.format(turn)}]
                category = query_chatgpt(client, message)
                model_classification.append({"note" : i, "utterance" : turn, "classifcation" : category})
                time.sleep(2)

        model_classification = pd.DataFrame(model_classification)

        with open(DATA_PATH.joinpath(f"{model_name}_{dataset}_c2.pkl"), 'wb') as f :
            pickle.dump(model_classification, f)

        send_line_message(f"C2 Evaluation for {model_name} and {dataset} is finished!")

    logging.debug("end of evaluation")

if __name__ == "__main__" :
    main()


# %%
