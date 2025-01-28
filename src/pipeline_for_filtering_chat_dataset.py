# This code is to develop a pipeline for filtering the chat dataset for Quality Control (QA)
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")