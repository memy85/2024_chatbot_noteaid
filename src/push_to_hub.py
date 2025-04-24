
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM, PeftModel, PeftConfig
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
MODEL_PATH = PROJECT_PATH.joinpath("model")

# model_path = "/home/wjang/2024_chatbot_noteaid/model/llama3.2-1B/chatbot7/checkpoint-3520"
model_path = "/home/wjang/2024_chatbot_noteaid/model/llama3.2-3B/chatbot0/checkpoint-7500"

# config = PeftConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model = PeftModel.from_pretrained(model, model_path)

# merge
merged = model.merge_and_unload()
merged.base_model.save_pretrained(MODEL_PATH.joinpath("merged/llama3.2-3B"))

# %%
tokenizer.push_to_hub("memy85/noteaid_chatbot_llama3.2-3B")
merged.base_model.push_to_hub("memy85/noteaid_chatbot_llama3.2-3B")


