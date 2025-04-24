import os,sys
from datasets import load_dataset, Dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import json
import argparse
import textstat
import openai
from openai import OpenAI

from utils import *


config = load_config()
filename = os.path.basename(__file__)
logging = config.load_logger(filename)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MAX_LENGTH = 60000
# checkpoint = 3330

# Load the trained model and tokenizer

models = {
    # "gpt-4o-mini": "",
    # "llama-3.2-3B": config.model_path("llama3.2-3B"),
    # "llama-3.2-3B-sft": PROJECT_PATH.joinpath("model/llama3.2-3B/chatbot14/checkpoint-100"),
    # "llama-3.2-3B-lora": PROJECT_PATH.joinpath("model/llama3.2-3B/chatbot13/checkpoint-200"),
    # "llama-3.2-3B-lora-ppo": "/home/htran/generation/med_preferences/AgentClinic/ppo_checkpoints/checkpoint-140",
    "llama-3.2-3B-lora-grpo": "/home/htran/generation/med_preferences/AgentClinic/grpo_checkpoints/checkpoint-latest",
}

# Generate predictions
def generate_response(tokenized_prompt, model, tokenizer, max_length=MAX_LENGTH):
    output_ids = model.generate(tokenized_prompt, max_new_tokens=200)
    arr_output = output_ids.detach().cpu().numpy()

    start_of_generate_index = tokenized_prompt.shape[1]
    response = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
    return response

def form_dataset() :
    with open(DATA_PATH.joinpath("synthetic_notes_for_test.pkl"), "rb") as f :
        notes = pickle.load(f)
    
    with open(DATA_PATH.joinpath("synthetic_conversation_10000.pkl"), "rb") as f :
        chat = pickle.load(f)

    prompt = config.load_prompts("autoeval_stage1.txt")
    
    new_rows = []
    for i, row in notes.iterrows() :
    #     # q = str(q)
        note = row['note']
        conversation = chat[i]
        system_prompt = prompt.format(discharge_note=note)

        new_rows.append({"messages" : [{"role" : "system", "content" : system_prompt}] + conversation })
    new_rows = Dataset.from_list(new_rows)
    return new_rows


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
            max_tokens=200,
        )
    except :
        logging.debug("API Call did not work..")

    return completion.choices[0].message.content

def evaluate_readability(predictions):
    readability_results = {}
    for pred in predictions:
        ease_score = textstat.flesch_reading_ease(pred)
        fk_score = textstat.flesch_kincaid_grade(pred)
        readability_results[pred] = {
            "flesch_reading_ease": ease_score,
            "flesch_kincaid_grade": fk_score
        }
    return readability_results

def compute_metrics(predictions, references):
    bleu = load("bleu")
    rouge = load("rouge")
    bertscore = load("bertscore")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {
        "bleu": bleu_score,
        "rouge": rouge_score,
        "bertscore": bertscore_result
    }


def arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    return args

def main() :
    args = arguments()
    TEST_FLAG = args.test
    test_data = form_dataset()

    for model_name, model_path in models.items() :
        logging.debug(f"model : {model_name}")

        if model_name == "gpt-4o-mini" :
            client = OpenAI(
                organization=os.getenv("OPENAI_ORG_ID"),
            )
            model = client

        elif model_name in ["llama-3.2-3B", "llama-3.2-3B-lora-ppo", "llama-3.2-3B-lora-grpo", "llama-3.2-3B-sft"]  :
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        elif model_name in ["llama-3.2-3B-lora"] :
            model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        else :
            continue

        logging.debug(f"successfully loaded model")


        # Evaluate responses
        predictions = []
        references = []

        print("The total length is : ", len(test_data))
        try :
            if TEST_FLAG :
                stopidx = 2
                stopcount = 0
            
            for idx, conversation in enumerate(test_data):

                # add system messages
                if TEST_FLAG :
                    if (stopcount == stopidx) :
                        with open(DATA_PATH.joinpath("measure_score_test.pkl"), "wb") as f :
                            pred_and_ref = (predictions,references)
                            pickle.dump(pred_and_ref, f)
                        logging.debug("successfully saved output")
                        break
                    else :
                        pass
                else :
                    pass

                # t = test_data[0]
                # t['messages']
                # first add system prompt
                messages = [conversation['messages'][0]]
                for turn in conversation['messages'] :
                    if turn['role'] == "user":
                        messages.append(turn)

                    elif turn['role'] == "assistant" :

                        # Generate response
                        if model_name == "gpt-4o-mini" :
                            assistant_response = query_chatgpt(client, messages)

                        else : 
                            tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
                            assistant_response = generate_response(tokenized_chat, model, tokenizer)

                        # append response to history
                        messages.append({"role":"assistant", "content":assistant_response})

                        # store predictions and ground truth
                        predictions.append(assistant_response)
                        references.append(turn['content'])  

                    else : 
                        continue

                if TEST_FLAG :
                    stopcount += 1

                if idx % 50 == 0 :
                    logging.debug(f"------------- finished {idx} --------------")
                    send_line_message(f"------------- finished {idx} --------------")
        except :

            send_line_message(f"Error: at idx {idx} the process met an error")
            logging.debug(f"Error: at idx {idx} the process met an error")
            sys.exit()


        metric_scores = compute_metrics(predictions, references)
        readability = evaluate_readability(predictions)
        metric_scores["readability"] = readability

        with open(DATA_PATH.joinpath(f"{model_name}_c1.pkl"), 'wb') as f :
            pickle.dump(metric_scores, f)

        send_line_message(f"Evaluation for {model_name} is finished!")

    logging.debug("end of evaluation")

if __name__ == "__main__" :
    main()


# %%
