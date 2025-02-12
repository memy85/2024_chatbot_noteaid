
#
# Case study code
# We study if the iteration pipeline works and checks the scores for case study
# Q. How do we know if this iteration should not go over sometime?
# Q. Can we make this process robust? i.e. How can we control this process? -> We set a maximum iteration count (3) and we take the last modified conversation 
#


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

MODEL_PATH = config.model_path("llama3.2-3B")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

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


def modification_call(conversation, evaluation) :
    '''
    conversation : changed to chat format
    evaluation : json format
    '''
    evaluation = str(evaluation)
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : modification_prompt.format(conversation=conversation, evaluation=evaluation)}
            ],
        )
        modified_conversation = completion.choices[0].message.content
        modified_conversation = modified_conversation.replace("```json", "")
        modified_conversation = modified_conversation.replace("```", "")
    except :
        logging.debug("API Call did not work..")

    try : 
        # we try to change it into a json format
       
        modified_conversation = json.loads(modified_conversation)
        return modified_conversation
    except :
        logging.debug("json conversion failed")
        # print(modified_conversation)
        raise RuntimeError("json conversion failed")


def check_scores(scores) :
    # scores is already json format
    strategy_contents = scores.keys()
    score_list = []
    for strategy in strategy_contents :
        val = int(scores[strategy]['score'])
        if val < 4 : score_list.append(False)
        else : score_list.append(True)
    return not all(score_list)


def iteration_pipeline(original_conversation) :
    '''
    original conversation : list of dictionary format
    '''
    conversation = original_conversation
    flag = True
    iteration_max = 3
    iteration_cnt = 0
    while flag :
        # change to conversation format
        formated = format_to_new_messages(conversation)
        scores = strategy_evaluation(formated)
        if iteration_cnt == 0 :
            initial_scores = scores
        # update flag
        flag = check_scores(scores)
        if iteration_max == iteration_cnt :
            break
        if flag :
            # update the conversation 
            conversation = modification_call(formated, scores)
        else :
            # if the flag changed, we escape
            flag = False
        iteration_cnt += 1
    return conversation, scores, initial_scores

        
def format_to_new_messages(chats) : 
    # new_system_prompt = "You are a helpful assistant trained for healthcare."
    # x["messages"][0]["content"] = new_system_prompt
    text = ""
    for chat in chats : 
        if chat['role'] == "system" :
            continue
        else :
            text += f"{chat['role']} : {chat['content']}\n"
    # new_message = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=False)

    return text


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
        # we use chatgpt-4o-mini
        pass

    else :
        model = pipeline("text-generation", model=MODEL_PATH, device_map="auto")

    dataset = load_datasets()
    # dataset['train'][0]['messages']

    categories_for_dataset = []

    # ** Description ** #
    # This code is for case-study. 
    # Therefore, we will only use index 2,3 to see the outputs
    case_study = {} 
    index = [2,3]
    for i in index : 
        # i = 2
        # data_type = 'train'
        case_study[i] = {}
        # conversation = dataset[data_type]['new_messages'][i]
        conversation = dataset[data_type][i]['messages']
        # change the system prompt due to privacy regulations
        conversation[0]['content'] = "You are a helpful assistant trained for healthcare."
        case_study[i]['messages'] = conversation

        if openai_flag :
            # raw_eval = strategy_evaluation(conversation)
            
            # ** Explanation ** 
            # the input of iteration pipeline is the conversation that is parsed
            # problem is we need to also parse for every modification 

            modified_conv, modified_scores, initial_scores = iteration_pipeline(conversation)

            time.sleep(2)
        else :
            out = model(prompt.format(conversation), return_full_text=False, max_new_tokens=6)

        case_study[i]['initial_scores'] = initial_scores 
        case_study[i]['modified_scores'] = modified_scores
        case_study[i]['modified_conv'] = modified_conv
 

    if openai_flag : 
        with open(DATA_PATH.joinpath(f"evaluation_case_strategy_gpt4o_{data_type}.pkl"), 'wb') as f :
            pickle.dump(case_study, f)
    else :  
        with open(DATA_PATH.joinpath(f"evaluation_case_strategy_{data_type}.pkl"), 'wb') as f :
            pickle.dump(case_study, f)

# hell

