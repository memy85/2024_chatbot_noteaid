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
import pandas as pd
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/raw")

# %%

whole_dataset = []
for page in ["page1", "page2", "page3", "page4"] :
    data = df = pd.read_excel(DATA_PATH.joinpath("annotation_notes.xlsx"), sheet_name=page)
    whole_dataset.append(data)

df = pd.concat(whole_dataset, ignore_index=True)
df.columns

df = df[["note_id", "text", "questionnaire"]].dropna()

# %%
# working on some filtering
df.loc[:,"note_id"] = df['note_id'].str.replace("\n", "")

example = df["questionnaire"][0]
print(example)

# %%
# define function that parses each question and choices
import json
import re

def parse_questions(questionnaire_text) :
    question_lists = []
    questions = example.split("|\n")
    # p = re.compile();
    for q in questions :
        idx = q.index("?")
        question = re.findall(r"^\d+\.\s*(.*)",q)[0]
        # print(question)
        choices = re.findall(r"([a-c])\)\s*(.*?)(?:\s*\((.*?)\))?\n", q)
        # print(choices)
        output = {"question" : question, 
                  "choices" {choice[0]: choice[1].strip() for choice in choices},
                  "answer": list(filter(lambda x : x[2] == "Answer", choices))[0][0]}
        question_lists.append(output)
    return question_lists












