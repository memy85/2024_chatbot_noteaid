from openai import OpenAI
import openai
import json
import re
import pickle
import time
import os, sys
import argparse
from utils import *

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts/instructions.txt")

client = OpenAI(
    organization=os.getenv("OPENAI_ORG_ID"),
)

def query_chatgpt(instruction, ehr_text) :
    try :
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role" : "system", "content" : "You are a helpful assistant trained for healthcare-related text processing"},
                {"role" : "user", "content" : f"### Instruction: {instruction}"},
                {"role" : "user", "content" : f"### discharge EHR Text: {ehr_text}"},
            ]

        )
    except :
        print("API Call did not work..", file=sys.stderr)

    return completion.choices[0].message.content

def reformat_outputs(output) :
    messages = [{"role": "system", "content":"You are a helpful assistant trained for healthcare-related text processing"}]
    pattern = re.compile(r"^(doctor|patient):\s*(.+)$", re.MULTILINE)
    for match in pattern.finditer(output) :
        role, content = match.groups()
        messages.append({"role": role, "content": content.strip()})
    return messages

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="choose between mimic and pittsburgh")
    parser.add_argument("--save_name", type=str)
    args = parser.parse_args()
    return args
    

def main() :
    args = parse_args()
    dataset = args.dataset 
    save_name = args.save_name

    dataset_path = config.load_discharge_notes(dataset)
    df = pd.read_pickle(dataset_path)
    df.reset_index(drop=True, inplace=True)
    
    with open(PROMPT_PATH, 'r') as f :
        instruction = f.read()

    # ===== FOR TESTING ===== 
    # ehr_text = df.texts[0]
    
    # output = query_chatgpt(instruction, ehr_text)
    # reformatted_output = reformat_outputs(output)
    # print(reformatted_output)

    results = []
    try :
        for idx, row in df.iterrows():

                ehr_text = row.texts
                
                output = query_chatgpt(instruction, ehr_text)
                reformatted_output = reformat_outputs(output)
                results.append({"messages" : reformatted_output})
                print(reformatted_output, file=sys.stderr)
                
                if idx == 1000 :
                    print("\n\n", file=sys.stderr)
                    print("===================== "*4, file=sys.stderr)
                    print(f"finished {idx}", file=sys.stderr)
                    print("===================== "*4, file=sys.stderr)
                    print("\n\n", file=sys.stderr)

                time.sleep(1)
        df.loc["reformatted_output"] = results

    except :
        print("stopped the process..", file=sys.stderr)
        df['reformatted_output'] = pd.NA
        df.loc[:idx,"reformatted_output"] = results

        # if idx == 2 :
        #     break

    with open(DATA_PATH.joinpath(f"{save_name}_results.pkl"), "wb") as f :
        pickle.dump(results, f)

    with open(DATA_PATH.joinpath(f"{save_name}.pkl"), "wb") as f :
        pickle.dump(df, f)




if __name__ == "__main__" :
    main()
