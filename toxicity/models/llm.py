import constants
from constants import ENV_VARS
from judges.judge import is_valid_answer
from llmapi.custom_model_api import Llm, get_llm
from toxicity.toxicity import ToxicityEvaluator, BooleanToxicityEvaluatorWrapper
import utils.util
from utils.stats import timing
import logging

logger = logging.getLogger(__name__)


def set_up_logging():
    f_format = logging.Formatter('%(message)s')
    file_name = f"{ENV_VARS['LOG_BASE']}/logs/current_llm_output.log"
    f_handler = logging.FileHandler(file_name)
    f_handler.setFormatter(f_format)
    f_handler.setLevel(logging.DEBUG)

    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)


if constants.LOGGING_ENABLED:
    set_up_logging()

class LlmModel(BooleanToxicityEvaluatorWrapper):
    def __init__(self, llm_id):
        self._llm = get_llm(llm_id=llm_id)
        self.BATCH_SIZE = 2

    def predict(self, sentence: str) -> float:
        output_sentence = self._llm.generate(sentence)
        # print("output_sentence", output_sentence)
        # note that this gives a measure of toxicity.
        return 0 if is_valid_answer(output_sentence) else 1

    @timing(name="BT_LLM")
    def predict_batch(self, sentences: list[str]):
        print("Trying to batch predict >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", flush=True)
        results = []
        for batched_sentences in utils.util.split_batch(sentences, self.BATCH_SIZE):
            semi_res = self._llm.batch_generate(batched_sentences)
            for _input, output in zip(batched_sentences, semi_res):
                logger.info(f"{_input}$#${output}")

            results.extend(semi_res)
            print("batch predicted", flush=True)
        print("batch predict successful <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", flush=True)
        return [0 if is_valid_answer(res) else 1 for res in results]
