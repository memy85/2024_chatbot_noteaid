# This code is to develop a pipeline for filtering the chat dataset for Quality Control (QA)
import openai
import pickle
import json

from utils import *

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
