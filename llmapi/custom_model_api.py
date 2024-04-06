from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum

from constants import ENV_VARS, FREE_CUDA_ID, FREE_LLM_CUDA_ID
from transformers import pipeline
from utils import util


class LlmId(Enum):
    VICUNA_7B = "lmsys/vicuna-7b-v1.5"
    GEMMA_2B_IT = "google/gemma-2b-it"


class Llm:
    MODELS_DIR = ENV_VARS["MODELS_DIR"]

    def __init__(self, llm_id: LlmId):
        self.modelId = llm_id
        self.device = FREE_LLM_CUDA_ID
        # Use a pipeline as a high-level helper
        directory = f"{Llm.MODELS_DIR}/{self.modelId.value}"
        print(directory)
        self.pipe = pipeline("text-generation", model=directory, device=self.device)

    def generate(self, input_str: str):
        pass




class Vicuna(Llm):
    def __init__(self):
        super(Vicuna, self).__init__(LlmId.VICUNA_7B)

    def generate(self, input_str: str):
        SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        USER_PROMPT = f"{{ {SYSTEM_PROMPT} }} USER: {{ {input_str} }} ASSISTANT:"
        # PARAMS =
        output = self.pipe(USER_PROMPT, do_sample=True)[0]['generated_text'].split("ASSISTANT:")[1].strip()
        return output


class Gemma(Llm):
    def __init__(self):
        super(Gemma, self).__init__(LlmId.GEMMA_2B_IT)

    def generate(self, input_str: str):
        input_ids = self.tokenizer(input_str, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, max_length=1000)
        return self.tokenizer.decode(outputs[0]).replace(input_str, "", 1).replace("<bos>", "").replace("<eos>", "").strip(
            " .?\t\s\n")


def get_llm(llm_id: LlmId) -> Llm:
    print(llm_id)
    print(llm_id == LlmId.GEMMA_2B_IT)
    print(llm_id == LlmId.VICUNA_7B)
    if llm_id == LlmId.GEMMA_2B_IT:
        return Gemma()
    elif llm_id == LlmId.VICUNA_7B:
        return Vicuna()
    else:
        raise Exception()
