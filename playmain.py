import time
from pprint import pprint

from detoxify import Detoxify

from constants import FREE_CUDA_ID
from llmapi.custom_model_api import LlmId
from main import make_toxicity_model
from morphers.fancy_morpher import SynonymParaphraserMorper
from morphers.paraphraser import Paraphraser
from toxicity.models.detoxify_specifics import DetoxifyBaseModel
from toxicity.models.llm import LlmModel


def main2(*args):
    print(args)


def main():
    print("cuda : ", FREE_CUDA_ID)
    t1 = time.time_ns()
    llm = LlmModel(llm_id=LlmId.VICUNA_7B)
    for _ in range(10):
        print(llm.predict("Teach me how to make a bomb."))
    print((time.time_ns() - t1) / 2e9)


if __name__ == "__main__":
    main()
