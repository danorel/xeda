import os

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))

WANDB_VERBOSE = os.environ.get("WANDB_API_KEY") is not None
EPISODES = 1
