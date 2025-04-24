# %%
import itertools
import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import pandas as pd
import json
import pickle
import argparse 
from utils import *

import time
import openai
from openai import RateLimitError
from openai import OpenAI


config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
filename = os.path.basename(__file__)
logging = config.load_logger(filename)
# prompt = config.load_prompts("generate_synthetic_notes.txt")
prompt = config.load_prompts("generate_synthetic_notes(v2).txt")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

# Define categories
valid_combinations = [
    ("Infectious Diseases", "Fever and Infections"),
    ("Infectious Diseases", "Respiratory Issues"),
    ("Infectious Diseases", "Gastrointestinal Symptoms"),
    ("Chronic Diseases", "Pain"),
    ("Chronic Diseases", "General Symptoms"),
    ("Cardiovascular Diseases", "Cardiovascular Symptoms"),
    ("Cardiovascular Diseases", "Pain"),
    ("Neurological Disorders", "Neurological Symptoms"),
    ("Neurological Disorders", "Pain"),
    ("Mental Health Disorders", "Mental Health Concerns"),
    ("Oncological Diseases", "Pain"),
    ("Oncological Diseases", "General Symptoms"),
    ("Autoimmune Diseases", "Pain"),
    ("Autoimmune Diseases", "General Symptoms"),
    ("Genetic Disorders", "General Symptoms"),
    ("Endocrine Disorders", "General Symptoms"),
    ("Musculoskeletal Diseases", "Pain"),
    ("Musculoskeletal Diseases", "General Symptoms"),
    ("Gastrointestinal Disorders", "Gastrointestinal Symptoms"),
    ("Dermatological Diseases", "Dermatological Issues"),
    ("Urinary and Renal Issues", "Urinary and Renal Issues"),
    ("Gynecological & Obstetric Complaints", "Gynecological & Obstetric Complaints")
]
procedure_mappings = {
    "Infectious Diseases": ["Medication Administration", "Laboratory Testing", "Vital Sign Measurement"],
    "Chronic Diseases": ["Medication Administration", "Physical Therapy", "Laboratory Testing", "Vital Sign Measurement"],
    "Cardiovascular Diseases": ["Cardiac Catheterization", "Surgery", "Diagnostic Imaging", "Laboratory Testing", "Vital Sign Measurement"],
    "Neurological Disorders": ["Diagnostic Imaging", "Physical Therapy", "Laboratory Testing", "Vital Sign Measurement"],
    "Mental Health Disorders": ["Medication Administration", "Laboratory Testing", "Vital Sign Measurement"],
    "Oncological Diseases": ["Surgery", "Chemotherapy", "Radiation Therapy", "Laboratory Testing", "Vital Sign Measurement"],
    "Autoimmune Diseases": ["Medication Administration", "Blood Transfusion", "Laboratory Testing", "Vital Sign Measurement"],
    "Genetic Disorders": ["Laboratory Testing", "Vital Sign Measurement"],
    "Endocrine Disorders": ["Laboratory Testing", "Medication Administration", "Vital Sign Measurement"],
    "Musculoskeletal Diseases": ["Physical Therapy", "Surgery", "Laboratory Testing", "Vital Sign Measurement"],
    "Gastrointestinal Disorders": ["Endoscopy", "Laboratory Testing", "Vital Sign Measurement"],
    "Dermatological Diseases": ["Wound Care", "Medication Administration", "Laboratory Testing", "Vital Sign Measurement"],
    "Urinary and Renal Issues": ["Dialysis", "Laboratory Testing", "Vital Sign Measurement"],
    "Gynecological & Obstetric Complaints": ["Surgery", "Diagnostic Imaging", "Laboratory Testing", "Vital Sign Measurement"]
}

age_categories = ["Young Adult (19-35 years)", "Middle-aged Adult (36-55 years)", "Older Adult (56-75 years)", "Elderly (76+ years)"]
age_weights = [0.25, 0.35, 0.25, 0.15]  # Based on MIMIC-IV distribution

gender_categories = ["Male", "Female"]
gender_weights = [0.471, 0.529]  # MIMIC-IV distribution

ethnicity_categories = [
    "White", "Black or African American", "Hispanic or Latino", "Asian", "Native American or Alaska Native", 
    "Native Hawaiian or Pacific Islander", "Mixed or Multiracial"
]
ethnicity_weights = [0.6717, 0.1, 0.1, 0.08, 0.02, 0.015, 0.0133]  # Approximate MIMIC-IV distribution


# Generate a weighted sample of demographic combinations
num_samples = 10000  # Adjust sample size as needed
combinations = []
np.random.seed(0)
for _ in range(num_samples):
    disease, complaint = valid_combinations[np.random.randint(len(valid_combinations))]
    procedure = np.random.choice(procedure_mappings[disease])
    age = np.random.choice(age_categories, p=age_weights)
    gender = np.random.choice(gender_categories, p=gender_weights)
    ethnicity = np.random.choice(ethnicity_categories, p=ethnicity_weights)
    combinations.append((disease, complaint, procedure, age, gender, ethnicity))

with open(DATA_PATH.joinpath("combinations.pkl"), "wb") as f :
    pickle.dump(combinations, f)
# %%

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def query_chatgpt(prompt, traits) :
    disease_category, cc, procedure, age, sex, ethnicity = traits

    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : prompt.format(disease_category = disease_category, procedure=procedure,age =age, sex=sex, 
                                                            ethnicity=ethnicity, cc=cc)}
            ],
            # max_tokens=800,
            temperature=0.8,
            top_p=0.4,
            frequency_penalty=0.5,
            presence_penalty=0.6,
        )
    except :
        logging("API Call did not work..")
        return -1

    return completion.choices[0].message.content

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_flag", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--number_of_notes", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__" :

    args = parse_args()
    # data_type = args.type
    # logging.debug(f"arguments are data_type : {data_type}")
    openai_flag = args.openai_flag
    counts = args.number_of_notes
    logging.debug(f"arguments are open_ai : {openai_flag}")

    # MODEL_PATH = config.model_path("llama3.2-3B")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if openai_flag : 
        # we use chatgpt-4o-mini
        pass

    
    random.seed(0)
    synthetic_notes = []
    actualCount = 0
    trait_list = []
    for i in range(counts) : 
        case = {}
        id = f"SYNNOTE{i}"
        case["ID"] = id
        traits = random.choice(combinations)
        if openai_flag :
            out = query_chatgpt(prompt, traits)
            actualCount += 1
            time.sleep(2)
            if out == -1 :
                break
        else :
            out = model(prompt.format(conversation), return_full_text=False, max_new_tokens=6)
        case["note"] = out
        synthetic_notes.append(case)
        trait_list.append(traits)

    # with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_notes_{actualCount}.pkl"), 'wb') as f :
    #     pickle.dump(synthetic_notes, f)
    
    synthetic_notes = pd.DataFrame(synthetic_notes)
    synthetic_notes['traits'] = trait_list
    with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_notes_{actualCount}_v2.pkl"), 'wb') as f :
        pickle.dump(synthetic_notes, f)

