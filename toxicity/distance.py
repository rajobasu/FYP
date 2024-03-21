import math

from morphers.fancy_morpher import RandomMorpher, Morpher
from toxicity.toxicity import ToxicityEvaluator
from utils.stats import GLBL_STATS, timing


@timing(name="GET_DIST")
def get_distance(toxicity_rater: ToxicityEvaluator, sentence: str) -> float:
    random_morpher = RandomMorpher()

    sentences = [sentence]
    LIMIT = 30
    for _ in range(LIMIT):
        sentences.append(random_morpher.modify(sentences[-1]))

    if toxicity_rater.predict(sentences[-1]) == 1:
        return 1

    lo = 0
    hi = 30
    while lo < hi:
        mid = int(math.ceil((lo + hi) / 2))
        if hi == mid:
            hi = hi - 1
            continue

        if toxicity_rater.predict(sentences[mid]) == 1:
            lo = mid + 1
        else:
            hi = mid

    return lo / LIMIT



def get_exponential_distance(toxicity_rater: ToxicityEvaluator, sentence: str):
    pass