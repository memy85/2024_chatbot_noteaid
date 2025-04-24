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
prompt = config.load_prompts("generate_qa_tests.txt")

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
    original_choices = ['a', 'b', 'c']
    new_choices = ['a', 'b', 'c']
    random.shuffle(new_choices)

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
    parser.add_argument("--number_of_notes", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__" :

    args = parse_args()
    number_of_notes = args.number_of_notes

    path = DATA_PATH.joinpath("gpt4o_mini_generated_synthetic_notes_10000.pkl")
    with open(path, 'rb') as f :
        synthetic_notes = pickle.load(f)
    
    qas = []
    for i, note in enumerate(synthetic_notes): 
        if i == number_of_notes :
            break
        p = prompt.format(discharge_note=note)
        out = query_chatgpt(p)
        time.sleep(2)
        if out == -1 :
            break

        qas.append(out)

    with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_qa_{i}.pkl"), 'wb') as f :
        pickle.dump(qas, f)

    # post-processing
    sqas = []
    for qa in qas :
        qa = json.loads(qa)
        new_qa = []
        for q in qa :
            q = shuffler(q)
            new_qa.append(q)
        sqas.append(new_qa)

    with open(DATA_PATH.joinpath(f"synthetic_qa_{i}.pkl"), 'wb') as f :
        pickle.dump(sqas, f)




