import os, sys
import pandas as pd

import openai
from openai import RateLimitError
from openai import OpenAI
import json

from utils import *

config = load_config()
filename = os.path.basename(__file__)
logging = config.load_logger(filename)
extract_information_prompt = config.load_prompts("extract_information_from_synthetic_note.txt")

client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

def query_chatgpt(prompt) :
    # disease_category, cc, procedure, age, sex, ethnicity = traits

    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "user", "content" : prompt}, 
            ],
            # max_tokens=800,
        )
    except :
        logging("API Call did not work..")
        return -1

    return completion.choices[0].message.content


def main() :
    with open(DATA_PATH.joinpath(f"gpt4o_mini_generated_synthetic_notes_10000.pkl"), 'rb') as f :
        notes = pickle.load(f)
    
    output = []
    count = 0
    for i, note in enumerate(notes) : 
        
        prompt = extract_information_prompt.format(note)
        out = query_chatgpt(prompt)

        if out == -1 :
            break
        # if i == 2 :
        #     break
        output.append(out)
        count += 1

    with open(DATA_PATH.joinpath(f"synthetic_note_demographic_info_{count}.pkl"), "wb") as f :
        pickle.dump(output, f)
    
    try : 
        analysis_outputs = []
        for out in outs :
            parsed_out = json.loads(out)
            analysis_outputs.append(parsed_out)

        df = pd.DataFrame(analysis_outputs)
        df.to_pickle(DATA_PATH.joinpath("synthetic_note_demographic_df.pkl"))
    except :
        logging.debug("Failed to convert to json")

    



    pass

if __name__ == "__main__" :
    main()
