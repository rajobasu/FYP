from detoxify import Detoxify

from constants import FREE_CUDA_ID
from toxicity.toxicity import ToxicityEvaluator, EnsembleToxicityEvaluator
from utils.stats import timing

TOXIC_THRESHOLD = 0.05
ENSEMBLE_TOXIC_THRESHOLD = [TOXIC_THRESHOLD, 0.2, 0.5]


class DetoxifyModel(ToxicityEvaluator):
    def __init__(self):
        self.model = Detoxify('original', device=FREE_CUDA_ID)

    @timing("DTXF")
    def predict(self, sentence: str) -> float:
        prediction = self.model.predict(sentence)
        worst = max([item[1] for item in prediction.items()])
        return 1 if worst > TOXIC_THRESHOLD else 0

    def backdoor_predict(self, sentence: str) -> float:
        prediction = self.model.predict(sentence)
        worst = max([item[1] for item in prediction.items()])
        return worst


class DetoxifyEnsembleModel(EnsembleToxicityEvaluator):
    def __init__(self):
        self.model = Detoxify('original', device=FREE_CUDA_ID)

    def predict(self, sentence: str) -> list[int]:
        prediction = self.model.predict(sentence)
        worst = max([item[1] for item in prediction.items()])
        return [1 if worst > item else 0 for item in ENSEMBLE_TOXIC_THRESHOLD]

    def backdoor_predict(self, sentence: str) -> float:
        prediction = self.model.predict(sentence)
        worst = max([item[1] for item in prediction.items()])
        return worst
