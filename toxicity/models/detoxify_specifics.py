from detoxify import Detoxify

from constants import FREE_CUDA_ID
from toxicity.toxicity import ToxicityEvaluator
from utils.stats import timing

TOXIC_THRESHOLD = 0.05


class DetoxifyModel(ToxicityEvaluator):
    def __init__(self):
        self.model = Detoxify('original', device=FREE_CUDA_ID)

    @timing("DTXF")
    def predict(self, sentence: str) -> float:
        prediction = self.model.predict(sentence)
        worst = max([item[1] for item in prediction.items()])
        return 1 if worst > TOXIC_THRESHOLD else 0

