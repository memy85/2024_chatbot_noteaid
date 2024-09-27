import pickle
import pandas as pd
from pathlib import Path
import os, sys
import yaml

CURRENT_FILE_PATH = Path(__file__).absolute()

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
    
    


def load_config() :
    project_path = CURRENT_FILE_PATH.parents[1]
    config_path = project_path.joinpath("config/config.yaml")

    with open(config_path) as f : 
        config = yaml.load(f, yaml.SafeLoader)

    return Config(config)
