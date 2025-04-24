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

example = df["questionnaire"][34]
print(example)

# %%
# define function that parses each question and choices
import json
import re

def parse_questions(row) :
    questionnaire_text = row['questionnaire']
    # preprocessing
    questionnaire_text = questionnaire_text.strip()
    if "\n" == questionnaire_text[-2:] :
        questionnaire_text = questionnaire_text[:-2]

    question_lists = []
    questions = questionnaire_text.split("|\n")
    # p = re.compile();
    for idx, q in enumerate(questions) :
        # idx = q.index("?")
        # print("this is idx: ", idx)
        question = re.findall(r"^\d+[\.,\)]\s*(.*)",q)[0]
        print(question)
        choices = re.findall(r"([a-c])\)\s*(.*?)(?:\s*\((.*?)\))", q)
        # print(choices)
        output = {"question" : question, 
                  "choices": {choice[0]: choice[1].strip() for choice in choices}, 
                  "answer": list(filter(lambda x : str.lower(x[2]) == "answer", choices))[0][0]}
        question_lists.append(output)

    return question_lists

# %%
# find the issue row
for i, row in df.iterrows() : 
    try  : 
        parse_questions(row)
    except :
        raise RuntimeError(f"The error is caused in : {i} / {row['note_id']}")

# %%
# Now start parsing
p = df.apply(parse_questions, axis=1)
len(p[1][1]['choices'])

# %%
# create checker 
def checker(list_of_questions) :
    for q in list_of_questions :
        if len(q['choices']) < 3 :
            return True
    return False    

# %%
# identity problematic rows
for i, lq in enumerate(p) :
    a = checker(lq)
    if a :
        print(i)

# %%
p[0]

# %% 
# create shuffler
import random
random.seed(0)

def shuffler(question) :
    original_choices = ['a', 'b', 'c']
    new_choices = ['a', 'b', 'c']
    random.shuffle(new_choices)

    choice = question['answer']
    shuffled_choices = {}
    for oc, nc in zip(original_choices, new_choices) :
        shuffled_choices[oc] = question['choices'][nc]
        if nc == choice : 
            new_answer = oc

    question['choices'] = shuffled_choices
    question['answer'] = new_answer
    return question

s = p[0][0]

shuffler(s)

# %%
a = p.apply(lambda x: list(map(shuffler, x)))

df['original_extracted'] = p
df['question_rl'] = a


# %% 
# Save the dataset
df.to_pickle(PROJECT_PATH.joinpath('data/processed/rl_dataset.pkl'))


# %%
idx = 1
r = df.loc[idx, :]
r.note_id

# %%


# %% Read pickle
df = pd.read_pickle(PROJECT_PATH.joinpath('data/processed/rl_dataset.pkl'))
import json
a = json.loads(df.to_json())
a['note_id'].keys()


# %%
with open(PROJECT_PATH.joinpath('data/processed/synthetic_notes_for_rl.pkl'), 'rb') as f :
    rl = pickle.load(f)


with open(PROJECT_PATH.joinpath('data/processed/synthetic_notes_for_test.pkl'), 'rb') as f :
    test = pickle.load(f)

# %% Prepare gold label dataset

gold = pd.read_pickle(PROJECT_PATH.joinpath("data/processed/gold.pkl"))
gold.to_json(PROJECT_PATH.joinpath('data/processed/gold.json'))








