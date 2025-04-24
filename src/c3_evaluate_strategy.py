# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from datasets import load_dataset
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

models = [
    # "gpt-4o-mini", #done
    "llama3.2-3B", 
    # "llama3.2-3B-lora", #done
    # "llama3.2-3B-sft", #done
    "llama3.2-3B-lora-ppo", 
    "llama3.2-3B-lora-grpo"
    ]

strategy_evaluation_prompt = config.load_prompts("strategy_evaluation.txt")

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


def strategy_evaluation(conversation_text) :
    try :
        completion = client.chat.completions.create(
            model = 'gpt-4o-mini',
            messages = [
                {"role" : "user", 
                 "content" : strategy_evaluation_prompt.format(conversation_text)}
            ],
        )
        output = completion.choices[0].message.content
    except :
        logging.debug("API Call did not work..")
        raise RuntimeError("API call did not work")
        

    # if there is some other things that we don't need
    output = output.replace("```json", "")
    output = output.replace("```", "")
    # Now we convert this to a json format
    # And then return it
    try : 
        # print(output)
        output = json.loads(output)
        return output
    except :
        raise RuntimeError("failed to convert json")



def load_conversation(model, dataset_type) :

    with open(DATA_PATH.joinpath(f"evaluation_conversation_history_{model}_{dataset_type}.pkl"), "rb") as f :
        data = pickle.load(f)
   
    histories = []
    for case in data :
        conversation_history = ""
        for turn in case['conversation'] :
            if turn['role'] == "system" :
                continue
            conversation_history += f"{turn['role']} : {turn['content']}\n"
        histories.append(conversation_history)
    return histories


def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default='', type=str)
    parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=False)
    
    # parser.add_argument("--openai_flag", type=str2bool, nargs='?', const=True, default=False)
    return parser.parse_args()

if __name__ == "__main__" :

    args = parse_args()
    dataset_type = args.dataset_type
    test_flag = args.test
    logging.debug(f"arguments are data_type : {dataset_type}")

    if test_flag :
        conversations = load_conversation(models[0], dataset_type)
        evaluation_outputs = []
        for i, conv in enumerate(conversations) :
            if i == 3 :
                break
            output = strategy_evaluation(conv) 
            evaluation_outputs.append(output)

            with open(DATA_PATH.joinpath(f"evaluation_strategy_test.pkl"), 'wb') as f :
                pickle.dump(evaluation_outputs, f)
    else :
        for model_name in models :
            
            # model_name = "llama3.2-3B"
            # dataset_type = "gold"
            conversations = load_conversation(model_name, dataset_type)
            evaluation_outputs = []
            for conv in conversations :
                output = strategy_evaluation(conv) 
                evaluation_outputs.append(output)
                time.sleep(1)

            with open(DATA_PATH.joinpath(f"evaluation_strategy_{model_name}_{dataset_type}.pkl"), 'wb') as f :
                pickle.dump(evaluation_outputs, f)
            
            logging.debug(f"finished {model_name} {dataset_type}")

