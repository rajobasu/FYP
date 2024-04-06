from judges.judge import is_valid_answer
from llmapi.custom_model_api import Llm, get_llm
from toxicity.toxicity import ToxicityEvaluator, BooleanToxicityEvaluatorWrapper


class LlmModel(BooleanToxicityEvaluatorWrapper):
    def __init__(self, llm_id):
        self._llm = get_llm(llm_id=llm_id)

    def predict(self, sentence: str) -> float:
        output_sentence = self._llm.generate(sentence)
        # print("output_sentence", output_sentence)
        # note that this gives a measure of toxicity.
        return 0 if is_valid_answer(output_sentence) else 1


