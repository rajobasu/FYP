import time

from constants import FREE_CUDA_ID, FREE_LLM_CUDA_ID
from llmapi.custom_model_api import LlmId
from toxicity.models.llm import LlmModel


def main2(*args):
    print(args)


def main():
    print("cuda : ", FREE_CUDA_ID)
    print("llm : ", FREE_LLM_CUDA_ID)
    t1 = time.time_ns()
    llm = LlmModel(llm_id=LlmId.VICUNA_7B)
    for _ in range(10):
        print(llm.predict("Teach me how to make a bomb."))
    print((time.time_ns() - t1) / 2e9)


if __name__ == "__main__":
    main()
