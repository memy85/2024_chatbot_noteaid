from copy import deepcopy
import pandas as pd
import json
import pickle
import argparse 
from utils import *
import time
import random
import openai
from openai import RateLimitError
from openai import OpenAI


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
filename = os.path.basename(__file__)
logging = config.load_logger(filename)
prompt = config.load_prompts("refine_qa.txt")

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

def shuffler(question) :
    original_choices = list(question["choices"].keys())
    new_choices = deepcopy(original_choices)
    # new_choices = ['a', 'b', 'c']
    random.shuffle(new_choices)
    
    if isinstance(question["answer"], list) :
        choice = question["answer"][0]
    else :
        choice = question['answer']
    shuffled_choices = {}
    for oc, nc in zip(original_choices, new_choices) :
        shuffled_choices[oc] = question['choices'][nc]
        if nc == choice : 
            new_answer = oc

    question['choices'] = shuffled_choices
    question['answer'] = new_answer
    return question

def query_chatgpt(prompt) :
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
    parser.add_argument("--number_of_qas", type=int, default=2)
    return parser.parse_args()

if __name__ == "__main__" :

    args = parse_args()
    number_of_qas = args.number_of_qas

    path = DATA_PATH.joinpath("gpt4o_mini_generated_synthetic_qa_10000.pkl")
    with open(path, 'rb') as f :
        qas = pickle.load(f)
    
    converted_qas = []
    count = 0
    for i, qa in enumerate(qas): 
        if count == number_of_qas :
            break
        try : 
            converted_qa = json.loads(qa)
        except :
            p = prompt.format(qa)
            out = query_chatgpt(p)
            time.sleep(2)
            try : 
                converted_qa = json.loads(out)
            except :
                logging.debug(f"problematic idx : {i}")
                logging.debug(f"converted qa : {converted_qa}")
                exit()

        else :
            new_qs = []
            for q in converted_qa :
                if len(q["answer"]) > 1 :
                    logging.debug(f"conversion idx : {i}")
                    logging.debug(f"Converting for q : {q}")
                    continue
                else :
                    logging.debug(f"Converting for q : {q}")
                    new_q = shuffler(q)
                    new_qs.append(new_q)
            converted_qa = new_qs

        converted_qas.append(converted_qa)
        count += 1

    with open(DATA_PATH.joinpath(f"synthetic_qa_{count}.pkl"), 'wb') as f :
        pickle.dump(converted_qas, f)




