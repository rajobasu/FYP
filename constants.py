# these only set up BROADER PROGRAM PARAMETERS.
# HYPER PARAMETERS FOR OTHER THINGS IN THE PROGRAMARE SET ON THE INDIVIDUAL FILES FOR EXAMPLE IT HE SEARCH
import os

import numpy as np
from dotenv import load_dotenv


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- HELPER FUNCTIONS --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return f"cuda:{np.argmax(memory_available)}"


def load_env_file():
    load_dotenv(".env")
    return {
        "MODELS_DIR": os.getenv("MODELS_DIR"),
        "HUGGING_FACE_TOKEN": os.getenv("HUGGING_FACE_TOKEN")
    }


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- CONSTANTS --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


DEBUG_MODE = True
LOGGING_ENABLED = True
RECORD_EXPERIMENT = False

FREE_CUDA_ID = get_freer_gpu()
ENV_VARS = load_env_file()
