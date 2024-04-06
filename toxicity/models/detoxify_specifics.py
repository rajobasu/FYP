from detoxify import Detoxify

from constants import FREE_CUDA_ID
from toxicity.toxicity import ToxicityEvaluator, BooleanToxicityEvaluatorWrapper
from utils.stats import timing

TOXIC_THRESHOLD = 0.05
ENSEMBLE_TOXIC_THRESHOLD = [TOXIC_THRESHOLD, 0.2, 0.5]


class DetoxifyBaseModel(ToxicityEvaluator):
    def __init__(self):
        self.model = Detoxify('original', device=FREE_CUDA_ID)

    def predict(self, sentence: str) -> float:
        prediction = self.model.predict(sentence)
        worst = max([item[1] for item in prediction.items()])
        return worst


class DetoxifyBooleanWrapper(BooleanToxicityEvaluatorWrapper):
    def __init__(self, base_model: ToxicityEvaluator, threshold=TOXIC_THRESHOLD):
        self.model = base_model
        self.threshold = threshold

    def predict(self, sentence: str):
        # the base version is to have only one threshold. we can however modify to get results at more thresholds.
        worst = self.backdoor_predict(sentence)
        return 1 if worst > self.threshold else 0

    def backdoor_predict(self, sentence: str) -> float:
        return self.model.predict(sentence)
