from abc import ABC, abstractmethod

import numpy as np


class ToxicityEvaluator(ABC):
    @abstractmethod
    def predict(self, sentence: str) -> float:
        pass

    def stats(self) -> tuple[float, float]:
        return 0, 0


class ToxicityModelWrapper(ToxicityEvaluator):
    def __init__(self, model, distance_func):
        self._model = model
        self._distance_func = distance_func

    def predict(self, sentence: str) -> float:
        ans = self._distance_func(self._model, sentence)
        return ans

    def stats(self) -> tuple[float, float]:
        return float(np.average(self._model.time_list)), float(np.std(self._model.time_list))


