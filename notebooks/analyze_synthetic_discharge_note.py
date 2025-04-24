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
# start
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
counts = 10000

# version 1. using discharge note content criteria
# with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_notes_{counts}.pkl"), 'rb') as f :
#     samples = pickle.load(f)

# version 2. using QNOTE criteria
with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_notes_{counts}_v2.pkl"), 'rb') as f :
    samples = pickle.load(f)

# %% [markdown]
# Now we read the samples.

# %%
# for version 1 this is a list
# for version 2, this is a dataframe
print(samples.loc[0, "note"])


# %%
df = pd.DataFrame(samples)
print(df.loc[0,'note'])

# %%
sample = df.loc[10, 'note']

# %%
import re

def extract_gender(text) :
    out = re.findall("Sex: (\w+)", text)
    return out[0]

def extract_cc(text) :
    out = re.findall("Chief Complaint:\s*(.+)", text)
    return out[0]
    


# %%
sample = df.loc[3, 'note']
extract_gender(sample)
extract_cc(sample)

# %%
import pickle
import pandas as pd
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

count = 10000
with open(DATA_PATH.joinpath(f"synthetic_note_demographic_info_{count}.pkl"), "rb") as f :
    out = pickle.load(f)

# %%
print(out[60])

# %%
import json

outs = []
for i, d in enumerate(out):
    print(i)
    if "json" in d :
        d = d.replace("```", "")
        d = d.replace("json", "")
    if "python" in d :
        d = d.replace("```", "")
        d = d.replace("python", "")
    try : 
        o = json.loads(d)
    except :
        d = d.replace("'", '"')
        o = json.loads(d)
    outs.append(o)

# %%
d = out[36].replace("'", '"')
json.loads(d)

# %%
df = pd.DataFrame(outs)

# %%
df.loc[:, "Chief Complaint"] = df["Chief Complaint"].fillna(df["Chief Complaints"])

# %%
import matplotlib
import seaborn as sns
# matplotlib.use("agg")
import matplotlib.pyplot as plt

df['Chief Complaint'].hist()
plt.show()

# %%
df.columns

# %%
df["Sex"] = df["Sex"].replace("Female", "female")
df["Sex"] = df["Sex"].replace("Male", "male")
df.Sex.hist()

# %%
df["Age"].unique()
df["Age"] = df["Age"].replace("Elderly", "Elderly (76+ years)")
df["Age"] = df["Age"].replace("Young Adult", "Young Adult (19-35 years)")
df["Age"] = df["Age"].replace("Middle-aged-Adult", "Middle-aged-Adult (36-55 years)")
df["Age"] = df["Age"].replace("Older Adult", "Older Adult (56-75 years)")

#%%
df["Ethnicity"] = df["Ethnicity"].replace("", "Unknown")
df["Ethnicity"] = df["Ethnicity"].replace("Not specified", "Unknown")

#%%
df["Disease Categories"].unique()


# %%
print(out[312])
json.loads(out[36])


# %%
outs

# %%
df = pd.DataFrame(outs)


# %%
df = pd.read_pickle(DATA_PATH.joinpath("synthetic_note_demographic_df.pkl"))


# %%
import matplotlib
import seaborn as sns
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

with open(DATA_PATH.joinpath("combinations.pkl"), "rb") as f :
    combinations = pickle.load(f)


#%%

def combination_to_row(tuple_data) :
    return {"disease" : tuple_data[0] ,
    "complaint" : tuple_data[1],
    "procedure" : tuple_data[2],
    "age" : tuple_data[3],
    "gender" : tuple_data[4],
    "ethnicity" : tuple_data[5],
    }

# %%
data = list(map(combination_to_row, combinations))
df = pd.DataFrame(data)



# %%
import numpy as np
# plt.rc('font', size = 8)
fig = plt.figure(figsize=(12,16))

sns.set_theme(style="white", context="talk")

ax1 = plt.subplot(4,2,1)
# ax1.tick_params(axis="x", rotation=90)
sns.countplot(ax=ax1, data=df, y="age", palette="flare", order = ["Elderly (76+ years)","Older Adult (56-75 years)", "Middle-aged Adult (36-55 years)", "Young Adult (19-35 years)"])
ax1.set_title("Age")
ax1.set_ylabel("")

ax2 = plt.subplot(4,2,2)
sns.countplot(ax=ax2, data=df, y="gender", palette="flare")
ax2.set_title("Gender")
ax2.set_ylabel("")

ax3 = plt.subplot(4,2,(3,4))
sns.countplot(ax=ax3, data=df, y="ethnicity", palette="flare")
ax3.set_title("Ethnicity")
ax3.set_ylabel("")

ax4 = plt.subplot(4,2, (5,8))
sns.countplot(ax=ax4, data=df, y="disease", palette="flare")

ax4.set_title("Disease Category")
ax4.set_ylabel("")

# plt.subplots_adjust(hspace=2)
fig.suptitle("Demographic Information of Synthetic Notes")
plt.tight_layout()
plt.show()

# %%


# %%
import numpy as np
# plt.rc('font', size = 8)
fig = plt.figure(figsize=(12,16))

sns.set_theme(style="white", context="talk")

ax1 = plt.subplot(6,2,1)
# ax1.tick_params(axis="x", rotation=90)
sns.countplot(ax=ax1, data=df, y="age", palette="flare", order = ["Elderly (76+ years)","Older Adult (56-75 years)", "Middle-aged Adult (36-55 years)", "Young Adult (19-35 years)"])
ax1.set_title("Age")
ax1.set_ylabel("")

ax2 = plt.subplot(6,2,2)
sns.countplot(ax=ax2, data=df, y="gender", palette="flare")
ax2.set_title("Gender")
ax2.set_ylabel("")

ax3 = plt.subplot(6,2,(3,4))
sns.countplot(ax=ax3, data=df, y="ethnicity", palette="flare")
ax3.set_title("Ethnicity")
ax3.set_ylabel("")

ax4 = plt.subplot(6,2, (5,8))
sns.countplot(ax=ax4, data=df, y="disease", palette="flare")

ax4.set_title("Disease Category")
ax4.set_ylabel("")

ax5 = plt.subplot(6,2, (9,12))
sns.countplot(ax=ax5, data=df, y="complaint", palette="flare")

ax5.set_title("Chief Complaint Category")
ax5.set_ylabel("")

# plt.subplots_adjust(hspace=2)
fig.suptitle("Demographic Information of Synthetic Notes")
plt.tight_layout()
plt.show()

# %%
# Applying T-SNE for embedding
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
# AutoModelForCausalLM.from_pretrai

# %%
lengths = np.array([len(s['note']) for s in samples])
lengths

# %%
# The length of the discharge note
df = pd.DataFrame(samples)
textLength = df['note'].apply(lambda x : len(x))

textLength.mean()
textLength.std()


# %%
df.sample(50, random_state=0)




# %%
# Extract the demographic informations for each cases

print(out[40])


# %%
# Analyze new discharge note (v2)
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
counts = 2

with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_notes_{counts}_v2.pkl"), 'rb') as f :
    samples = pickle.load(f)

# %%
print(samples['note'][1])




# %%













