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
prompt = config.load_prompts("gpt_evaluation.txt")

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
            max_tokens=300,
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
    dataset['train']['new_messages'][0]

    categories_for_dataset = []
    scores = {} 

    for idx, conversation in enumerate(dataset[data_type]['new_messages']) :
        original_data = dataset[data_type]['messages'][idx]

        scores[idx]['messages'] = original_data

        if openai_flag :
            out = query_chatgpt(prompt, conversation)
            time.sleep(2)
        else :
            out = model(prompt.format(conversation), return_full_text=False, max_new_tokens=6)

        scores[idx]['output'] = out
 

    if openai_flag : 
        with open(DATA_PATH.joinpath(f"evaluation_score_strategy_gpt4o_{data_type}.pkl"), 'wb') as f :
            pickle.dump(case_study, f)
    else :  
        with open(DATA_PATH.joinpath(f"evaluation_score_strategy_{data_type}.pkl"), 'wb') as f :
            pickle.dump(case_study, f)
