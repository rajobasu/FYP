import os
from pprint import pprint
import numpy as np
from dotenv import load_dotenv
import random

random.seed(3324)
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print(f"cuda:{np.argmax(memory_available)}")
    return f"cuda:{np.argmax(memory_available)}"


def load_env_file():
    load_dotenv(".env")
    return {
        "MODELS_DIR": os.getenv("MODELS_DIR"),
        "HUGGING_FACE_TOKEN": os.getenv("HUGGING_FACE_TOKEN")
    }
