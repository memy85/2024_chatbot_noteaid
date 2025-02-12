#
#
#     This code aims to not only measure the score of the conversation dataset, but also attempts to improve the scores of the dataset.
#     (1) First the input conversation dataset is loaded 
#     (2) GPT-4o-mini evaluates the codes based on the given prompt.
#         (2.1) GPT-4o-mini outputs a score and evidences for those scores
#     (3) If the scores are not above or equal to 4, then we reiterate the process (2) until all the scores are above or equal to 4
#     (4) The final dataset is saved
#
#     After the process is finished we conduct a case study.
#
#

from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
from utils import *
import openai
from openai import OpenAI

config = load_config()
filename = os.path.basename(__file__)
logging = config.load_logger(filename)

PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
strategy_evaluation_prompt = config.load_prompts("strategy_evaluation.txt")
modification_prompt = config.load_prompts("conversation_modification_based_on_strategy.txt")


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

def strategy_evaluation(sentence) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : strategy_evaluation_prompt.format(sentence)}
            ],
            max_tokens=300,
            temperature=0.2
        )
    except :
        logging("API Call did not work..")

    return completion.choices[0].message.content

def modification_call(conversation, evaluation) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : modification_prompt.format(conversation=conversation, evaluation=evaluation)}
            ],
            # max_tokens=300,
            # temperature=0.2
        )
    except :
        logging("API Call did not work..")

    try : 
        modified_conversation = json.loads(completion.choices[0].message.content)
        return modified_conversation['messages']
    except :
        logging("json conversion failed")
        raise RuntimeError("json conversion failed")

def check_score(evaluation) :
    
    flag = False
    for category in [] :
        if int(evaluation[category]['score']) < 4 :
            return True
        else :
            pass
    return False

def get_score(evaluation) :
    '''
    based on the output of the evaluation, it parses and retrieves the scores.
    '''

    return 

            


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

    MODEL_PATH = config.model_path("llama3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if openai_flag : 
        # we use chatgpt-4o-mini
        pass

    else :
        model = pipeline("text-generation", model=MODEL_PATH, device_map="auto")

    dataset = load_datasets()
    
    def format_to_new_messages(x) : 
        new_system_prompt = "You are a helpful assistant trained for healthcare."
        x["messages"][0]["content"] = new_system_prompt
        new_message = tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)
        return new_message

    dataset = dataset.map(lambda x : {'new_messages': format_to_new_messages(x)})
    # dataset['train']['new_messages'][0]

    categories_for_dataset = []
    scores = {} 
    
    modified_conversations = []
    modified_evaluations = []
    raw_evaluations = []
    for idx, conversation in enumerate(dataset[data_type]['new_messages']) :

        flag = False
        original_data = dataset[data_type]['messages'][idx]
        scores[idx]['messages'] = original_data

        if openai_flag :
            raw_eval = strategy_evaluation(conversation)
            modified_conv = modification_call(conversation, raw_eval)

            tokenized_modified_conv = tokenizer.apply_chat_template(modified_conv, tokenize = False, add_generation_prompt=False)

            modified_eval = strategy_evaluation(tokenized_modified_conv)
            
            # check score
            flag = check_score(modified_eval)

            # We will leave here for further modificaiton
            # ----- Blank -----
            
            modified_conversations.append(modified_conv)
            modified_evaluations.append(modified_eval)
            raw_evaluations.append(raw_eval)

            time.sleep(2)
        else :
            out = model(prompt.format(conversation), return_full_text=False, max_new_tokens=6)

        # scores[idx]['raw_eval'] = raw_eval
        # scores[idx]['modified_conv'] = modified_conv
        # scores[idx]['modified_eval'] = modified_eval

    dataset[data_type].add_column("raw_eval", raw_evaluations)
    dataset[data_type].add_column("modified_eval", modified_evaluations)
    dataset[data_type].add_column("modified_conv", modified_conversations)
 

    if openai_flag : 
        with open(DATA_PATH.joinpath(f"evaluation_score_strategy_gpt4o_{data_type}.pkl"), 'wb') as f :
            pickle.dump(case_study, f)
    else :  
        with open(DATA_PATH.joinpath(f"evaluation_score_strategy_{data_type}.pkl"), 'wb') as f :
            pickle.dump(case_study, f)
