import random
from copy import deepcopy
import numpy as np
from nltk import sent_tokenize

from constants import RECORD_EXPERIMENT
from morphers.fancy_morpher import Morpher
from similarity.similarity import SentenceSimilarityModel
from storage.simple_storage import Storage
from toxicity.toxicity import ToxicityEvaluator
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
            print(sentencepool[0])
            print(item)
            raise Exception("Sentence lengths not equal")

    result_parts = []
    for i in range(length):
        result_parts.append(random.choice(sentencepool)[i])

    return result_parts


class EvoAlgoV1:
    def __init__(self,
                 toxicity_rater: ToxicityEvaluator,
                 modifier: Morpher,
                 similarity_rater: SentenceSimilarityModel,
                 db: Storage,
                 enable_crossover=True):
        self.NUM_CHILDREN = 5
        self.CROSSOVER = 50 if enable_crossover else 0
        self.POOL_SIZE = 20

        self.toxic = toxicity_rater
        self.sent_sim = similarity_rater
        self.modifier = modifier
        self.db = db
        self.generated_items = 0
        self.orig_sentence: str = ""  # note that this is treated as a single string since we dont really modify it anyway.

    def fitness(self, sentence: SENTENCE_T) -> FITNESS_T:
        return self.toxic.predict(" ".join(sentence)), self.sent_sim.predict(self.orig_sentence, " ".join(sentence))

    def mutate(self, sentence: SENTENCE_T) -> SENTENCE_T:
        return self.modifier.modify(sentence)

    def select(self, sentencepool: list[SENTENCE_INFO_T]):
        best_similarity = max([x[2] for x in sentencepool])

        def scoring(t: float, s: float, bs: float):
            return (1 - t) #* (s / bs)

        return sorted(sentencepool, key=lambda x: scoring(x[1], x[2], best_similarity), reverse=True)[:self.POOL_SIZE]


    def get_generation_as_batch(self, sentence):
        pass

    def create_generation(
            self, sentence: SENTENCE_T,
            generation_num: int,
    ) -> list[SENTENCE_INFO_T]:
        result: list[SENTENCE_INFO_T] = []

        for _ in range(self.NUM_CHILDREN):
            modified_sentence = self.mutate(sentence)
            toxic_score, similarity_score = self.fitness(modified_sentence)

            result.append((modified_sentence, toxic_score, similarity_score))
            debug_print(f"MUTATED: {toxic_score: .3f} {similarity_score : .3f} {modified_sentence} ")
            if RECORD_EXPERIMENT:
                self.db.add_record(" ".join(modified_sentence), toxic_score, similarity_score, generation_num)

            self.generated_items += 1
            print(f"\r{self.generated_items}", end="")

        return result

    def start_search(self, sentence: str, num_generations: int = 25):
        self.orig_sentence = sentence
        print("start search")

        tokenized_sentence: SENTENCE_T = sent_tokenize(sentence)
        toxicity_initial, similarity_initial = self.fitness(tokenized_sentence)
        sentence_pool: list[SENTENCE_INFO_T] = [(tokenized_sentence, toxicity_initial, similarity_initial)]
        for _ in range(num_generations):
            print(f"Generation {_}")

            # generate mutations of a sentence and add them to the pool.
            result = []
            for item in sentence_pool:
                result.extend(self.create_generation(item[0], _))

            self.generated_items = 0
            print()

            sentences_only = [x[0] for x in sentence_pool]
            for _ in range(self.CROSSOVER):
                sample = crossover(sentences_only)
                t, s = self.fitness(sample)
                debug_print(f"CROSSVR: {t} {s} {sample}")
                result.append((sample, t, s))

            sentence_pool.extend(result)
            print(f"min toxicity achieved  :{min([x[1] for x in sentence_pool])}")
            print(f"avg toxicity preselect :{np.mean([x[1] for x in sentence_pool])}")
            sentence_pool = self.select(sentence_pool)
            print(f"avg toxicity postselect:{np.mean([x[1] for x in sentence_pool])}")
