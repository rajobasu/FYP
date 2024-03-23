import math

from morphers.fancy_morpher import RandomMorpher
from toxicity.toxicity import ToxicityEvaluator
from utils.stats import timing

random_morpher = RandomMorpher()

@timing(name="GET_DIST")
def get_distance(toxicity_rater: ToxicityEvaluator, sentence: str) -> float:
    sentences = [sentence]
    LIMIT = 300
    for _ in range(LIMIT):
        sentences.append(random_morpher.modify(sentences[-1]))

    if toxicity_rater.predict(sentences[-1]) == 1:
        return 1

    lo = 0
    hi = 300
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