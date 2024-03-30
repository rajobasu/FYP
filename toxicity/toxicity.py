from abc import ABC, abstractmethod

import numpy as np


class ToxicityEvaluator(ABC):
    @abstractmethod
    def predict(self, sentence: str) -> int:
        pass

    def stats(self) -> tuple[float, float]:
        return 0, 0

    def set_params(self, params):
        pass


class EnsembleToxicityEvaluator(ABC):
    @abstractmethod
    def predict(self, sentence: str) -> list[int]:
        pass

    def stats(self) -> tuple[float, float]:
        return 0, 0

    def set_params(self, params):
        pass

class ToxicityModelWrapper(ToxicityEvaluator):
    def __init__(self, model, distance_func):
        self._model = model
        self._distance_func = distance_func
        self._distance_param = 300

    def predict(self, sentence: str) -> float:
        ans = self._distance_func(self._model, sentence, self._distance_param)
        return ans

    def stats(self) -> tuple[float, float]:
        return float(np.average(self._model.time_list)), float(np.std(self._model.time_list))

    def set_params(self, params):
        self._distance_param = params["distance_param"]


