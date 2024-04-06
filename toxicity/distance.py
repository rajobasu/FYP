import math

import numpy as np

from morphers.fancy_morpher import RandomMorpher
from toxicity.toxicity import ToxicityEvaluator, BooleanToxicityEvaluatorWrapper
from utils.stats import timing

random_morpher = RandomMorpher()


@timing(name="GET_DIST")
def binsrch(
        *,
        toxicity_rater: BooleanToxicityEvaluatorWrapper,
        limit: int,
        generated_sentences: list[str],
) -> float:
    if toxicity_rater.predict(generated_sentences[-1]) == 1:
        return 1

    lo = 0
    hi = limit
    while lo < hi:
        mid = int(math.ceil((lo + hi) / 2))
        if hi == mid:
            hi = hi - 1
            continue

        val = toxicity_rater.predict(generated_sentences[mid])
        if val == 1:
            lo = mid + 1
        else:
            hi = mid
    return lo / limit


# def get_distance_ensemble(evaluators: list[BooleanToxicityEvaluatorWrapper], sentence: str, limit) -> float:
#     vals = [binsrch(
#         toxicity_rater=evaluator,
#         limit=limit,
#         generated_sentences=generated_sentences
#     ) for evaluator in evaluators]
#
#     num_0 = vals.count(0.0)
#     return vals[-1] * (0.9 ** num_0)
#

class BinarySearcher:
    def __init__(self, *,
                 toxicity_rater: BooleanToxicityEvaluatorWrapper,
                 limit: int,
                 sentence: str):
        self.limit = limit
        self.generated_sentences = [sentence]
        for _ in range(limit):
            self.generated_sentences.append(random_morpher.modify(self.generated_sentences[-1]))

        self.lo = 0
        self.hi = limit

        # [1111100000] -> [lo, hi] should always contain a 1 or a 0. this is what we check below
        # [0000000000] -> we also check for this case explicitly
        # [1111111111] -> we check for this case explicitly

        if toxicity_rater.predict(self.generated_sentences[0]) == 0:
            self.lo = self.hi = 0
        elif toxicity_rater.predict(self.generated_sentences[-1]) == 1:
            self.lo = self.hi = limit
        self.mid = 0

    def get_next_question(self):
        if self.lo >= self.hi - 1:  # in effect if they have a gap of 2 or less
            return None

        self.mid = int(math.ceil((self.lo + self.hi) / 2))  # mid
        return self.generated_sentences[self.mid]

    def update_with_answer(self, answer):
        if answer is None:
            return

        if answer == 1:
            self.lo = self.mid
        elif answer == 0:
            self.hi = self.mid

    def get_answer(self):
        return self.lo / self.limit


def batch_distance(*, sentences: list[str], evaluator: BooleanToxicityEvaluatorWrapper, limit: int) -> list[float]:
    searchers = [BinarySearcher(toxicity_rater=evaluator, limit=limit, sentence=sentence) for sentence in sentences]
    while True:
        ids = []
        sentences = []
        for i, searcher in enumerate(searchers):
            sent = searcher.get_next_question()
            if sent is not None:
                ids.append(i)
                sentences.append(sent)

        if not ids:
            break

        results = evaluator.predict_batch(sentences)
        for i, res in zip(ids, results):
            searchers[i].update_with_answer(res)

    return [searcher.get_answer() for searcher in searchers]
