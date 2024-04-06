from judges.judge import is_valid_answer
from llmapi.custom_model_api import Llm, get_llm
from toxicity.toxicity import ToxicityEvaluator, BooleanToxicityEvaluatorWrapper
import utils.util
from utils.stats import timing


class LlmModel(BooleanToxicityEvaluatorWrapper):
    def __init__(self, llm_id):
        self._llm = get_llm(llm_id=llm_id)
        self.BATCH_SIZE = 8

    def predict(self, sentence: str) -> float:
        output_sentence = self._llm.generate(sentence)
        # print("output_sentence", output_sentence)
        # note that this gives a measure of toxicity.
        return 0 if is_valid_answer(output_sentence) else 1

    @timing(name="BT_LLM")
    def predict_batch(self, sentences: list[str]):
        print("Trying to batch predict >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        results = []
        for batched_sentences in utils.util.split_batch(sentences, self.BATCH_SIZE):
            results.extend(self._llm.batch_generate(batched_sentences))
        print("batch predict successful <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return [0 if is_valid_answer(res) else 0 for res in results]
