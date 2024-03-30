from abc import ABC, abstractmethod

import numpy as np


class ToxicityEvaluator(ABC):
    @abstractmethod
    def predict(self, sentence: str):
        pass

    def set_params(self, params):
        pass


class BooleanToxicityEvaluatorWrapper(ToxicityEvaluator, ABC):
    def __init__(self, base_model: ToxicityEvaluator, threshold):
        pass

    def backdoor_predict(self, sentence):
        pass


class ToxicityEnsembleModelWrapper:
    def __init__(self, distance_func):
        self._models = []
        self._distance_func = distance_func
        self._distance_param = 300

    def add_model(self, model: BooleanToxicityEvaluatorWrapper):
        self._models.append(model)

    def predict(self, sentence: str) -> float:
        ans = self._distance_func(self._models, sentence, self._distance_param)
        return ans

    def set_params(self, params):
        self._distance_param = params["distance_param"]

    def backdoor_predict(self, sentence):
        return self._models[-1].backdoor_predict(sentence)
