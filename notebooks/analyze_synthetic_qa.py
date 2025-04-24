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
from utils import *
import json
config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

# %%

with open(DATA_PATH.joinpath("gpt4o_mini_generated_synthetic_qa_10000.pkl"), "rb") as f :
    qas = pickle.load(f)

with open(DATA_PATH.joinpath("synthetic_qa_2.pkl"), "rb") as f :
    sqa = pickle.load(f)

# %%
import json
new_qas = []
for i, qa in enumerate(qas):
    print(i)
    new_qa = json.loads(qa)
    new_qas.append(new_qa)
# %%
json.loads(qas[0])

# %%
json.loads(qas[300])

# %%
txt = '[{"question" : "What is your main discharge diagnosis?", "choices" : { "a" : "Pain (UMLS C0030193)","b" : "Rheumatoid Arthritis (UMLS C0003868)", "c" : "Acute Bronchitis (UMLS C0005972)"},"answer" : "a"},{"question" : "What medication should you take daily post-discharge?","choices" : {"a" : "Ibuprofen 600 mg","b" : "Prednisone 10 mg","c" : "Aspirin 325 mg"},"answer" : "b"},{"question" : "What symptoms should you monitor for that require medical attent ion?","choices" : {"a" : "Fever above 100.4°F","b" : "Slight headache", "c" : "Occasional fatigue"},"answer" : "a"},{"question" : "Which activity is recommended post-discharge?","choices" : {"a" : "Gradual increase in physical activity","b" : "High-impact exercises","c" : "Complete bed rest"},"answer" : "a"},{"question" : "What secondary diagnosis do you have?","choices" : { "a" : "Systemic lupus erythematosus (UMLS C0022800)","b" : "Osteoarthritis (UMLS C0028760)","c" : "Chronic Fatigue Syndrome (UMLS C0012272)"},"answer" : "a" },{"question" : "When should you schedule your follow-up appointment?","choices" : {"a" : "Within one month","b" : "In six months","c" : "Next year"}, "answer" : "a"},{"question" : "What tests were performed during your hospital stay?","choices" : {"a" : "Serum laboratory tests with normal CBC","b" : "X-ray of the chest","c" : "MRI of the brain"},"answer" : "a"},{"question" : "What is a possible side effect of Prednisone?","choices" : {"a" : "Increased appetite","b" : "Dehydration","c" : "Constipation"},"answer" : "a"},{"question" : "How often can you take Ibuprofen for pain?","choices" : {"a" : "Every 8 hours as needed","b" : "Every 4 hours","c" : "Once a day"},"answer" : "a"},{"question" : "What is an indicator that you should return to the hospital?","choices" : {"a" : "Severe joint swelling","b" : "Mild discomfort in limbs","c" : "Slight increase in energy"},"answer" : "a"}]'
qas[232] = txt

# %%
qas[300]


txt = '[\n    {"question" : "What is your primary discharge diagnosis?",\n    "choices" : {\n        "a" : "Pain (UMLS: C155496)",\n        "b" : "Lupus",\n        "c" : "Coronary artery disease"},\n    "answer" : "a"\n    },\n    {"question" : "What medication is to be taken daily?",\n    "choices" : {\n        "a" : "Prednisone 10 mg",\n        "b" : "Ibuprofen 400 mg",\n        "c" : "Calcium Supplement 500 mg"},\n    "answer" : "a"\n    },\n    {"question" : "What should you do if you experience severe headaches?",\n    "choices" : {\n        "a" : "Contact the hospital/ED",\n        "b" : "Increase your ibuprofen dosage",\n        "c" : "Wait for 48 hours"},\n    "answer" : "a"\n    },\n    {"question" : "Which activity should you avoid post-discharge?",\n    "choices" : {\n        "a" : "Heavy lifting",\n        "b" : "Gentle stretching",\n        "c" : "Non-weight bearing exercises"},\n    "answer" : "a"\n    },\n    {"question" : "What laboratory tests were performed during your stay?",\n    "choices" : {\n        "a" : "CBC, CRP, and ESR",\n        "b" : "MRI and X-ray",\n        "c" : "Blood glucose and cholesterol"},\n    "answer" : "a"\n    },\n    {"question" : "When is your follow-up appointment scheduled?",\n    "choices" : {\n        "a" : "In 1 week",\n        "b" : "In 1 month",\n        "c" : "In 2 weeks"},\n    "answer" : "a"\n    },\n    {"question" : "What symptoms indicate you should return to the hospital?",\n    "choices" : {\n        "a" : "Chest pain and blurred vision",\n        "b" : "Mild headache and fatigue",\n        "c" : "Joint stiffness"},\n    "answer" : "a"\n    },\n    {"question" : "What type of medication is Methotrexate?",\n    "choices" : {\n        "a" : "An immunosuppressant",\n        "b" : "A pain reliever",\n        "c" : "An antibiotic"},\n    "answer" : "a"\n    },\n    {"question" : "What should you monitor for after discharge?",\n    "choices" : {\n        "a" : "Increased pain or swelling",\n        "b" : "Hair loss",\n        "c" : "Increased appetite"},\n    "answer" : "a"\n    },\n    {"question" : "What treatment provided significant symptom relief during your stay?",\n    "choices" : {\n        "a" : "Corticosteroid therapy",\n        "b" : "Antihypertensive medication",\n        "c" : "Antibiotic therapy"},\n    "answer" : "a"\n    }\n]'
#%%
qas[300] = txt


# %%
s = json.loads(qas[0])
list(s[0]['choices'].keys())
s


#%%
print(qas[2575])




# %%

with open(DATA_PATH.joinpath("synthetic_qa_10000.pkl"), "rb") as f :
    df = pickle.load(f)
df[1:5]

# %%
import pandas as pd
import json
import pickle
from utils import *
from datasets import Dataset, load_dataset
config = load_config()

PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

with open(DATA_PATH.joinpath("gpt4o_mini_generated_synthetic_notes_10000.pkl"), "rb") as f :
    notes = pickle.load(f)

with open(DATA_PATH.joinpath("synthetic_qa_10000.pkl"), "rb") as f :
    qa = pickle.load(f)

# %%
new_rows = []
for i, q in enumerate(qa) :
    notes[i]["qa"] = str(q)
    new_rows.append(notes[i])
df = pd.DataFrame(new_rows)
df = Dataset.from_pandas(df)

# %%
df.to_

# %%
notes[0].keys()

# %%
Dataset.from_pandas(df)

# %%
df.loc[0,"qa"]

#%%
df = pd.DataFrame(notes)
new_rows[0][1]

#%%

# %%
data[1:3]


# %%



