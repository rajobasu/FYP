# these only set up BROADER PROGRAM PARAMETERS.
# HYPER PARAMETERS FOR OTHER THINGS IN THE PROGRAMARE SET ON THE INDIVIDUAL FILES FOR EXAMPLE IT HE SEARCH
import os
from enum import Enum

import numpy as np
from dotenv import load_dotenv, dotenv_values


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- HELPER FUNCTIONS --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('rm -f tmp')
    return f"cuda:{np.argmax(memory_available)}"


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
ENV_VARS = dict(dotenv_values(".env"))
