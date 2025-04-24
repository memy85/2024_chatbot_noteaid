# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: cn
#     language: python
#     name: python3
# ---

# %%
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

# %%
import pandas as pd
import pickle

pitts = pd.read_pickle(DATA_PATH.joinpath("pittsburgh_chat.pkl"))
mimic = pd.read_pickle(DATA_PATH.joinpath("mimic_chat.pkl"))

# %%
pitts['data_source'] = "pitts"
mimic['data_source'] = "mimic"

# %%
pitts = pitts.rename(columns = {"reportID" : "note_id"})
pitts = pitts[['data_source', 'note_id', 'texts', 'reformatted_output']]

# %%
mimic = mimic[['data_source','note_id', 'texts', 'reformatted_output']]

# %%
pitts_and_mimic = pd.concat([pitts, mimic],ignore_index=True)

# %%
pitts.head()

# %%
pitts['reformatted_output'][1]


# %%
def format_input_text_for_training(row) :
    conversation = ""
    for turn in row['reformatted_output']['messages'] :
        if turn['role'] == 'system' :
            conversation += f"""You are a helpful assistant in healthcare. Here is the patient's discharge note. 
### Discharge note :
{row['texts']}
\n\n
            """
        conversation += f"{turn['role']} : {turn['content']}\n"
    return conversation


# %%
def format_for_sft(row) :
    discharge_note = row.texts
    conversation = row.reformatted_output
    if conversation['messages'][0]['role'] == 'system' :
        system_message = f"You are a helpful assistant trained for healthcare. Here is the patient's discharge note. \n\n {discharge_note}"
        conversation['messages'][0]['content'] = system_message
    for conv in row.reformatted_output['messages'] :
        if conv['role'] == 'doctor' :
            conv['role'] = 'assistant'
        elif conv['role'] == 'patient' :
            conv['role'] = 'user'

    return conversation


# %%
pitts_and_mimic['conversation'] = pitts_and_mimic.apply(format_for_sft,axis=1)


# %%
import json, jsonlines

conversation = pitts_and_mimic['conversation'].to_list()

with jsonlines.open(DATA_PATH.joinpath("conversation_dataset.jsonl"), 'w') as f :
    json.dump(conversation,f)

# %%
# test the saved dataset
with open(DATA_PATH.joinpath("conversation_dataset.json"), 'r') as f :
    conversation = json.load(f)

# %%
import random
random.seed(42)

random.shuffle(conversation)

# %%
idx = round(len(conversation) * 0.8)
train, test = conversation[:idx], conversation[idx:]

# %%
import jsonlines

# %%
with jsonlines.open(DATA_PATH.joinpath("train_conversation.jsonl"), 'w') as f :
    f.write(train)
with jsonlines.open(DATA_PATH.joinpath("test_conversation.jsonl"), 'w') as f :
    f.write(test)

# %%
test[2000]

# %%
pitts_and_mimic.to_pickle(DATA_PATH.joinpath("pitts_and_mimic.pkl"))

# %%
import pandas as pd

df = pd.read_pickle("../data/processed/pitts_and_mimic.pkl")

# %%
from datasets import Dataset, load_dataset

data = load_dataset("json", data_files=DATA_PATH.joinpath("test_conversation.jsonl").as_posix())

# %%
model_path = config.model_path('llama3.2-1B')
model, tokenizer = config.load_model(model_path)


# %%
df = data.map(lambda x : {"formatted_chat" : tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)})

# %%
print(df['conversation_text'][0])

# %%
train[0]

# %%
from datasets import load_dataset

data = load_dataset("philschmid/dolly-15k-oai-style",split="train")

# %%
from datasets import load_dataset
from utils import * 
config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

data = load_dataset("json", data_files={"train" : DATA_PATH.joinpath("train_conversation.jsonl").as_posix(),
                                    "test" : DATA_PATH.joinpath("test_conversation.jsonl").as_posix()})
