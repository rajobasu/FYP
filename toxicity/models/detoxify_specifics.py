from detoxify import Detoxify

from toxicity.toxicity import ToxicityEvaluator
from utils.stats import timing
from utils.util import FREE_CUDA_ID

TOXIC_THRESHOLD = 0.05


class DetoxifyModel(ToxicityEvaluator):
    def __init__(self):
        self.model = Detoxify('original', device=FREE_CUDA_ID)

    @timing("DTXF")
    def predict(self, sentence: str) -> float:
        prediction = self.model.predict(sentence)
        worst = max([item[1] for item in prediction.items()])
        return 1 if worst > TOXIC_THRESHOLD else 0
