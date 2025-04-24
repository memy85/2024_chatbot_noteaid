from datasets import load_dataset, Dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import json
import argparse
import textstat

from utils import *

import openai
from openai import OpenAI

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

# Load the trained model and tokenizer

models = {
    "gpt-4o-mini": client,
    "llama-3.2-3B": AutoModelForCausalLM.from_pretrained("YOUR_MODEL_PATH", device_map="auto"),
    "finetuned_model": AutoModelForCausalLM.from_pretrained("YOUR_MODEL_PATH", device_map="auto"),
    "finetuned_rl_model": AutoModelForCausalLM.from_pretrained("YOUR_MODEL_PATH", device_map="auto"),
    "rl_finetuned_model": AutoModelForCausalLM.from_pretrained("YOUR_MODEL_PATH", device_map="auto"),
}

# Tokenizer is removed for gpt-4o-mini
tokenizers = {
    model_name: AutoTokenizer.from_pretrained(model_name)
    for model_name in models.keys() if model_name != "gpt-4o-mini"
}

# def load_model(MODEL_PATH) :
#     model = AutoPeftModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     return model, tokenizer

# Generate predictions
def generate_response(tokenized_prompt, model, tokenizer,max_length=MAX_LENGTH):
    output_ids = model.generate(tokenized_prompt, max_new_tokens=100, num_beams=5)
    arr_output = output_ids.detach().cpu().numpy()

    start_of_generate_index = tokenized_prompt.shape[1]
    response = tokenizer.batch_decode(arr_output[:, start_of_generate_index:], skip_special_tokens=True)[0]
    return response

def form_dataset() :
    with open(DATA_PATH.joinpath("synthetic_notes_for_test.pkl"), "rb") as f :
        notes = pickle.load(f)
    
    with open(DATA_PATH.joinpath("synthetic_conversation_10000.pkl"), "rb") as f :
        chat = pickle.load(f)
        # sft_chat = []
        # for i in notes.index :
        #     sft_chat.append(chat[i])
        # chat = sft_chat

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

def query_chatgpt(message) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = message,
            max_tokens=100,
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
    # parser.add_argument("--model", type=str, default=None)
    # parser.add_argument("--trial", type=int, default=1)
    # parser.add_argument("--baseline", type=str2bool, nargs='?', const=True, default=False)
    # parser.add_argument("--openai_flag", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    return args

def main() :
    args = arguments()
    # MODEL_NAME = args.model
    # TRIAL = args.trial
    # BASELINE = args.baseline
    # OPENAI_FLAG = args.openai_flag
    TEST_FLAG = args.test

    test_data = form_dataset()

    for model_name, model in models.items() :
        if k == "gpt-4o-mini"
            model_name
        else :
            tokenizers[k]

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

            messages = [conversation['messages'][0]]

            for turn in conversation['messages'] :
                if turn['role'] == "user":
                    messages.append(turn)

                elif turn['role'] == "assistant" :

                    # Generate response
                    if OPENAI_FLAG :
                        assistant_response = query_chatgpt(messages)

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

            if idx % 1000 == 0 :
                print(f"------------- finished {idx} --------------")
                send_line_message(f"------------- finished {idx} --------------")
    except :

        send_line_message(f"Error: at idx {idx} the process met an error")
        logging.debug(f"Error: at idx {idx} the process met an error")
        sys.exit()

        # pred_and_ref = (predictions,references)
        # with open(DATA_PATH.joinpath("preds_and_references.pkl"), "wb") as f :
        #     pickle.dump(pred_and_ref, f)

        # with open(DATA_PATH.joinpath("preds_and_references.pkl"), "rb") as f :
        #     predictions, references = pickle.load(f)

    # Compute metrics
    # bleu = load("bleu")
    # rouge = load("rouge")
    # bertscore = load("bertscore")
    #
    # # BLEU
    # bleu_score = bleu.compute(predictions=predictions, references=references)
    # print(f"BLEU Score: {bleu_score}")
    #
    # # ROUGE
    # rouge_score = rouge.compute(predictions=predictions, references=references)
    # print(f"ROUGE Score: {rouge_score}")
    #
    # # BERTScore
    # bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    # print(f"BERTScore: Precision: {bertscore_result['precision']}, Recall: {bertscore_result['recall']}, F1: {bertscore_result['f1']}")
    #


    metric_scores = compute_metrics(predictions, references)
    readability = evaluate_readability(predictions)
    metric_scores["readability"] = readability

    if BASELINE :
        scores = {"model name" : f"{MODEL_NAME}",
                "bleu" : bleu_score,
                "rouge" : rouge_score, 
                "bertscore" : bertscore_result}

        with open(DATA_PATH.joinpath(f"{MODEL_NAME}_chatbot.pkl"), 'wb') as f :
            pickle.dump(scores, f)

    elif OPENAI_FLAG :
        scores = {"model name" : "gpt4o-mini",
                  "bleu" : bleu_score, 
                  "rouge" : rouge_score,
                  "bertscore" : bertscore_result}

        with open(DATA_PATH.joinpath(f"gpt4o_mini_chatbot.pkl"), 'wb') as f :
            pickle.dump(scores, f)
    else : 

        scores = {"model name" : f"{MODEL_NAME}-chatbot{TRIAL}-checkpoint-{checkpoint}",
                "bleu" : bleu_score,
                "rouge" : rouge_score, 
                "bertscore" : bertscore_result}

        with open(DATA_PATH.joinpath(f"{MODEL_NAME}_chatbot{TRIAL}.pkl"), 'wb') as f :
            pickle.dump(scores, f)

    send_line_message(f"Evaluation for {MODEL_NAME} chatbot{TRIAL} is finished!")

    logging.debug("end of evaluation")


if __name__ == "__main__" :
    main()


# %%
