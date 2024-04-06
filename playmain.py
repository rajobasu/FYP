import time

from constants import FREE_CUDA_ID, FREE_LLM_CUDA_ID
from dataparser import get_data
from llmapi.custom_model_api import LlmId
from morphers.fancy_morpher import RandomMorpher
from toxicity.distance import random_morpher
from toxicity.models.llm import LlmModel


def main2(*args):
    print(args)


def main():
    print("cuda : ", FREE_CUDA_ID)
    print("llm : ", FREE_LLM_CUDA_ID)
    t1 = time.time_ns()
    llm = LlmModel(llm_id=LlmId.VICUNA_7B)
    data = get_data()
    morpher = RandomMorpher()
    for _ in range(100):
        val = llm.predict(data)
        print(val, data)
        data = morpher.modify(data)


    print((time.time_ns() - t1) / 2e9)


if __name__ == "__main__":
    main()
