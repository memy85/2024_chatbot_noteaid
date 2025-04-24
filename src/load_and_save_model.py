import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PPO_PATH =  '/home/htran/generation/med_preferences/AgentClinic/ppo_checkpoints/checkpoint-140'
GRPO_PATH = '/home/htran/generation/med_preferences/AgentClinic/grpo_checkpoints/checkpoint-latest'

def arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--trial_number", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__" :
    args = arguments()
    model_type = args.model
    trial_number = args.trial_number
    checkpoint = args.checkpoint

    base_path = config.model_path(model_type)

    # MODEL PATH
    model_type = 'llama3.2-3B'
    trial_number = 14
    checkpoint = 100
    # base_path = config.model_path(model_type)
    MODEL_PATH = Path(f"/home/wjang/2024_chatbot_noteaid/model/{model_type}/chatbot{trial_number}/checkpoint-{checkpoint}")
    
    base_path = GRPO_PATH
    # base_path = PPO_PATH
    base_model = AutoModelForCausalLM.from_pretrained(base_path, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model.push_to_hub("memy85/chatbot_noteaid_ppo")
    tokenizer.push_to_hub("memy85/chatbot_noteaid_ppo")

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    merged_model = model.merge_and_unload()
    model.base_model
    
    model_type = 'grpo'
    trial_number = 1
    save_path = PROJECT_PATH.joinpath(f"model/gguf/{model_type}/chatbot{trial_number}")
    # save_path = PROJECT_PATH.joinpath(f"model/gguf/ppo/chatbot1")
    base_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # model.save_pretrained(save_path)
    # merged_model.base_model.save_pretrained(save_path)
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# %% Test
# model_type='llama3.2-3B'
# trial_number=11
# MODEL_PATH = PROJECT_PATH.joinpath(f"model/gguf/{model_type}/chatbot{trial_number}").as_posix()
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#
# input_ids = tokenizer("hello")['input_ids']
# base_model.generate(input_ids)

