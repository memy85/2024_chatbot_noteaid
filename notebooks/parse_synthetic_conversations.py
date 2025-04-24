# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import json
import pandas 
from utils import *
config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

# %%

with open(DATA_PATH.joinpath("gpt4o_mini_generated_synthetic_conversation_10000.pkl"), "rb") as f :
    conv = pickle.load(f)
# %%

print(conv[3]['messages'])

# %%
import openai
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

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

# %%
prompt = """
The given string has some errors and cannot be converted to json. Fix the error and give it back to me.
Please only output the json and no other words. Do not put any ``` or ```json or ```python in to the string. Just the json format itself.

Input : 
{}

Output : 
"""

# %%
problematic = []
conversation = []
for i, c in enumerate(conv) :
    out = c['messages']
    while True : 
        try : 
            m = json.loads(out)
            conversation.append(m[1:])
            break
        except :
            print(i)
            p = prompt.format(out)
            out = query_chatgpt(p)
            continue

    # remove system message

# %%

with open(DATA_PATH.joinpath("synthetic_conversation_10000.pkl"), "wb") as f :
    pickle.dump(conversation, f)


# %%

with open(DATA_PATH.joinpath("synthetic_conversation_10000.pkl"), "rb") as f :
    synthetic_conversation = pickle.load(f)

# %%
synthetic_conversation[0]




# %%
json.loads(c["messages"])

# %%
from datasets import load_dataset

def load_datasets() :

    data = load_dataset("json", data_files={"train" : DATA_PATH.joinpath("train_conversation.jsonl").as_posix(),
                                     "test" : DATA_PATH.joinpath("test_conversation.jsonl").as_posix()})
    return data

data = load_datasets()


data['train']['messages'][0]
