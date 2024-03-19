from detoxify import Detoxify

from toxicity.toxicity import ToxicityEvaluator

TOXIC_THRESHOLD = 0.05


class DetoxifyModel(ToxicityEvaluator):
    def __init__(self):
        self.model = Detoxify('original', device="cuda:0")

    def predict(self, sentence: str) -> float:
        prediction = self.model.predict(sentence)
        return 1 if prediction["toxicity"] > TOXIC_THRESHOLD else 0
