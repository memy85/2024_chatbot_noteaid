import argparse
from utils import *

config = load_config()
PROJECT_PATH = config.project_path
DATA_PATH = PROJECT_PATH.joinpath("data/processed")
FIGURE_PATH = PROJECT_PATH.joinpath("figures")


def arguments() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--")

    args = parser.parse_args()
    return args
    

def main() :

    pass

if __name__ == "__main__" :
    main()