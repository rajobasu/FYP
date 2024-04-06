from judges.judge import is_valid_answer
from llmapi.custom_model_api import Llm, get_llm
from toxicity.toxicity import ToxicityEvaluator


class LlmModel(ToxicityEvaluator):
    def __init__(self, llm_id):
        self._llm = get_llm(llm_id=llm_id)

    def predict(self, sentence: str) -> float:
        output_sentence = self._llm.generate(sentence)
        print("output_sentence", output_sentence)
        return 1 if is_valid_answer(output_sentence) else 0
