from abc import ABC, abstractmethod

import numpy as np


class ToxicityEvaluator(ABC):
    @abstractmethod
    def predict(self, sentence: str):
        pass

    def set_params(self, params):
        pass


class BooleanToxicityEvaluatorWrapper(ToxicityEvaluator, ABC):
    def backdoor_predict(self, sentence):
        return 0


class ToxicityEnsembleModelWrapper:
    def __init__(self, distance_func):
        self._models = []
        self._distance_func = distance_func
        self.distance_param = 256

    def add_model(self, model: BooleanToxicityEvaluatorWrapper):
        self._models.append(model)

    def predict(self, sentence: str) -> float:
        ans = self._distance_func(self._models, sentence, self.distance_param)
        return ans

    def set_params(self, params):
        self.distance_param = params["distance_param"]

    def backdoor_predict(self, sentence):
        return self._models[-1].backdoor_predict(sentence)

    def set_distance_param(self, dist):
        self.distance_param = dist
