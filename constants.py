# these only set up BROADER PROGRAM PARAMETERS.
# HYPER PARAMETERS FOR OTHER THINGS IN THE PROGRAMARE SET ON THE INDIVIDUAL FILES FOR EXAMPLE IT HE SEARCH
import os
from enum import Enum

import numpy as np
from dotenv import load_dotenv, dotenv_values


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- HELPER FUNCTIONS --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_freer_gpu(best=0):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    with open("tmp", "r") as f:
        mem_av = [(int(x.split()[2]), i) for i, x in enumerate(f.readlines())]
        answer = sorted(mem_av, reverse=True)[min(len(mem_av) - 1, best)][1]
    os.system('rm -f tmp')
    return f"cuda:{answer}"


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- CONSTANTS --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class ScoringMethods(Enum):
    FRONTIER = 0,
    REDUCER = 1


DEBUG_MODE = False
LOGGING_ENABLED = True
RECORD_EXPERIMENT = True

FREE_CUDA_ID = get_freer_gpu()
FREE_LLM_CUDA_ID = get_freer_gpu()
ENV_VARS = dict(dotenv_values(".env"))
