import random

import numpy as np
from nltk import sent_tokenize

from constants import ScoringMethods
from morphers.fancy_morpher import Morpher
from similarity.similarity import SentenceSimilarityModel
from storage.simple_storage import Storage
from toxicity.distance import batch_distance
from toxicity.toxicity import ToxicityEnsembleModelWrapper, BooleanToxicityEvaluatorWrapper
from utils.util import debug_print

"""
Note that all throughout here we treat a sentence as a list of strings. 
"""

SENTENCE_T = list[str]
FITNESS_T = tuple[float, float]
SENTENCE_INFO_T = tuple[SENTENCE_T, float, float]


def crossover(sentencepool: list[SENTENCE_T]) -> SENTENCE_T:
    length = len(sentencepool[0])
    for item in sentencepool:
        if len(item) != length:
            debug_print(sentencepool[0])
            debug_print(item)
            raise Exception("Sentence lengths not equal")

    result_parts = []
    for i in range(length):
        result_parts.append(random.choice(sentencepool)[i])

    return result_parts


class EvoAlgoV1:
    def __init__(self,
                 toxicity_rater: BooleanToxicityEvaluatorWrapper,
                 modifier: Morpher,
                 similarity_rater: SentenceSimilarityModel,
                 db: Storage,
                 search_params):
        self.NUM_CHILDREN = search_params["num_children"]
        self.CROSSOVER = search_params["crossover"]
        self.POOL_SIZE = search_params["pool_size"]
        self.SCORING_FUNC = search_params["scoring_func"]
        self.GROWTH_ADDITIVE = 0 if search_params["growth_delta"] is None else search_params["growth_delta"]
        self.current_pool_size = self.POOL_SIZE
        self.last_pool_min = 1.0
        self.selector = self.select_frontier \
            if search_params["scoring_method"] == ScoringMethods.FRONTIER \
            else self.select
        self.throw_half = search_params["throw_half"]
        self.auto_dist = search_params["auto_dist"]
        self.distance_param = search_params["distance_param"]

        if self.NUM_CHILDREN is None:
            raise Exception()
        if self.CROSSOVER is None:
            raise Exception()
        if self.POOL_SIZE is None:
            raise Exception()

        self.toxic = toxicity_rater
        self.sent_sim = similarity_rater
        self.modifier = modifier
        self.db = db
        self.generated_items = 0
        self.orig_sentence: str = ""  # note that this is treated as a single string since we dont really modify it anyway.

    def fitness(self, sentence: SENTENCE_T) -> FITNESS_T:
        return self.batch_fitness([sentence])[0]

    def batch_fitness(self, sentences: list[SENTENCE_T]) -> list[FITNESS_T]:
        converted_sentences = [" ".join(sentence) for sentence in sentences]
        toxicities = batch_distance(
            sentences=converted_sentences,
            evaluator=self.toxic,
            limit=self.distance_param
        )
        similarities = [self.sent_sim.predict(self.orig_sentence, sentence) for sentence in converted_sentences]

        return [(t, s) for t, s in zip(toxicities, similarities)]

    def mutate(self, sentence: SENTENCE_T) -> SENTENCE_T:
        return self.modifier.modify(sentence)

    def select(self, sentencepool: list[SENTENCE_INFO_T]):
        best_similarity = max([x[2] for x in sentencepool])
        min_tox = min([x[1] for x in sentencepool])

        def scoring1(t: float, s: float, bs: float):
            return (1 - t) * (s / bs)

        def scoring2(t: float, s: float, bs: float):
            return (1 - t) ** 2 / (s / bs)

        def scoring3(t: float, s: float, bs: float):
            return (1 - t) / ((s / bs) ** 2)

        scoring = [scoring1, scoring2, scoring3][self.SCORING_FUNC]

        selected_list = sorted(sentencepool, key=lambda x: scoring(x[1], x[2], best_similarity), reverse=True)
        selected_list = selected_list[:2 * self.current_pool_size]
        selected_list = sorted(selected_list, key=lambda x: x[1])

        if min_tox == self.last_pool_min:
            self.current_pool_size += self.GROWTH_ADDITIVE
        elif self.current_pool_size > self.POOL_SIZE:
            self.current_pool_size -= self.GROWTH_ADDITIVE

        self.last_pool_min = min_tox

        return selected_list[:self.current_pool_size]

    def select_frontier(self, sentencepool: list[SENTENCE_INFO_T]):
        sorted_sentences = sorted(sentencepool, key=lambda x: (x[1], -x[2]))  # increasing toxicity
        # we now create a frontier
        result_sentences = []
        best_similarity = 0.0
        for sentence in sorted_sentences:
            if sentence[2] >= best_similarity:
                result_sentences.append(sentence)
                best_similarity = sentence[2]

        return result_sentences[:self.current_pool_size]

    def start_search(self, sentence: str, num_generations: int = 40):
        self.orig_sentence = sentence
        debug_print("start search")

        tokenized_sentence: SENTENCE_T = sent_tokenize(sentence)

        toxicity_initial, similarity_initial = self.fitness(tokenized_sentence)
        sentence_pool: list[SENTENCE_INFO_T] = [(tokenized_sentence, toxicity_initial, similarity_initial)]
        for ng in range(num_generations):
            print(f"Generation {ng: >3}", end="")

            # generate mutations of a sentence and add them to the pool.
            result = []

            sentence_only_pool = [item[0] for item in sentence_pool]

            outputs = self.modifier.modify_batch(sentence_only_pool, children_per_sentence=self.NUM_CHILDREN)
            for _ in range(self.CROSSOVER):
                outputs.append(crossover(sentence_only_pool))

            fitness_vals = self.batch_fitness(outputs)
            for output, (t, s) in zip(outputs, fitness_vals):
                result.append((output, t, s))

            self.generated_items = 0

            denom = self.distance_param
            for sent, t, s in result:
                sent_str = " ".join(sent)
                self.db.add_record(t * denom, self.toxic.backdoor_predict(sent_str), s, ng)  # type: ignore

            if self.throw_half:
                sentence_pool = sentence_pool[:self.current_pool_size // 2]

            if self.auto_dist:
                mean_toxicity = np.mean([x[1] for x in result])
                self.distance_param = int(1.3 * denom * mean_toxicity + 4)

            sentence_pool.extend(result)
            print(f" min :{min([x[1] for x in sentence_pool]) * denom:.5f}", end="")
            print(f" avg pre:{np.mean([x[1] for x in sentence_pool]) * denom:.5f}", end="")
            sentence_pool = self.selector(sentence_pool)
            print(f" avg post:{np.mean([x[1] for x in sentence_pool]) * denom:.5f}")

            # we need to re-evaluate the sentences if auto_dist is turned on
            if self.auto_dist:
                sentence_pool = [(sent, *self.fitness(sent)) for sent, _, _ in sentence_pool]
