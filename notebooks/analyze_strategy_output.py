
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
import pandas as pd
import numpy as np
import json
import pickle

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

# %%
results = pd.read_pickle(DATA_PATH.joinpath("evaluation_case_strategy_gpt4o_train.pkl"))
output = results[3]
output.keys()

output['initial_scores']
output['modified_scores']
conversation = output['messages']
t = output['modified_conv']


def format_conv(conversation): 
    text = ""
    for conv in conversation :
        utterance = conv['content'] role = conv['role']
        text += f"{role}: {utterance}\n"
    print(text)

format_conv(output['modified_conv'])



# %%
# parse the output to json

out = json.loads(output)
out

out['messages']


# %%
# format conversation 

results[2]['messages']






# %% [markdown]
# # Testing matplotlib
# - I don't think this is working...?

# %%
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.show()


# %%
from datasets import load_dataset

def load_datasets() :

    data = load_dataset("json", data_files={"train" : DATA_PATH.joinpath("train_conversation.jsonl").as_posix(),
                                     "test" : DATA_PATH.joinpath("test_conversation.jsonl").as_posix()})
    return data

# %%

d = load_datasets()

d['train'].add_column("new", [0 for _ in range(len(d['train']))])
d['train']


