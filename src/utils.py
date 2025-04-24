import pickle
import requests
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, sys
import random
import logging
import yaml

CURRENT_FILE_PATH = Path(__file__).absolute()
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

special_tokens_dict = {
    "pad_token": DEFAULT_PAD_TOKEN,
    "eos_token": DEFAULT_EOS_TOKEN,
    "bos_token": DEFAULT_BOS_TOKEN,
    "unk_token": DEFAULT_UNK_TOKEN,
}


class Config :

    def __init__(self, config_file) :
        self.file = config_file

    @property 
    def project_path(self) :
        return Path(self.file['project_path'])

    def model_path(self, model_name) :
        return self.file['model_path'][model_name]

    @property 
    def device(self) :
        return self.file['device']

    @property 
    def representative_notes(self) :
        project_path = self.project_path
        notes_path = project_path.joinpath('data/raw/representative_notes_50_new.xlsx')
        print(notes_path)
        df = pd.read_excel(notes_path)
        return df
    
    def load_discharge_notes(self, name) :
        project_path = self.project_path
        if name == "pittsburgh" :
            dataset_name = self.file['data']['pittsburgh']
            return project_path.joinpath(f"data/processed/{dataset_name}")
        elif name == "mimic" :
            dataset_name = self.file['data']['mimic']
            return project_path.joinpath(f"data/processed/{dataset_name}")
    
    def load_model(self, MODEL_PATH) :
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.add_special_tokens({
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                })
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    
    def load_prompts(self, prompt_name) :
        path = self.project_path.joinpath(f"prompts/{prompt_name}")
        with open(path, 'r') as f :
            prompt =  f.read()
        return prompt
    
    def load_logger(self, filename) :
        id = random.randrange(0,10000)
        filepath = self.project_path.joinpath(f'logs/{filename}_{id}.log').as_posix()
        print(f"logging at {filepath} / id : {id}")
        logging.basicConfig(filename=filepath,level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging 


def load_config() :
    project_path = CURRENT_FILE_PATH.parents[1]
    config_path = project_path.joinpath("config/config.yaml")

    with open(config_path) as f : 
        config = yaml.load(f, yaml.SafeLoader)

    return Config(config)


def send_line_message(message):
    # LINE Messaging API endpoint
    url = 'https://api.line.me/v2/bot/message/push'

    # Request headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}'
    }

    # Request body
    data = {
        'to': LINE_USER_ID,
        'messages': [
            {
                'type': 'text',
                'text': message
            }
        ]
    }
    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print(f'Failed to send message. Status code: {response.status_code}, Response: {response.text}')
