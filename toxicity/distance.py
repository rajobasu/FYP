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


def get_distance_ensemble(evaluators: list[BooleanToxicityEvaluatorWrapper], sentence: str, limit) -> list[float]:
    generated_sentences = [sentence]
    for _ in range(limit):
        generated_sentences.append(random_morpher.modify(generated_sentences[-1]))

    vals = [binsrch(
        toxicity_rater=evaluator,
        limit=limit,
        generated_sentences=generated_sentences
    ) for evaluator in evaluators]

    weights = [x + 1 for x in range(len(vals))]
    weights.reverse()
    return np.dot(vals, weights) / sum(weights)


def get_exponential_distance(toxicity_rater: ToxicityEvaluator, sentence: str):
    pass
