import random
from copy import deepcopy

from nltk import sent_tokenize

from morphers.fancy_morpher import Morpher
from similarity.similarity import SentenceSimilarityModel
from storage.simple_storage import Storage
from toxicity.toxicity import ToxicityEvaluator

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
                 db: Storage):
        self.MIN_NUM_CHILDREN = 10
        self.MIN_CROSSOVER = 50
        self.MAX_POOL_SIZE = 100

        self.toxic = toxicity_rater
        self.sent_sim = similarity_rater
        self.modifier = modifier
        self.db = db
        self.orig_sentence: str = ""  # note that this is treated as a single string since we dont really modify it anyway.

    def fitness(self, sentence: SENTENCE_T) -> FITNESS_T:
        return self.toxic.predict(" ".join(sentence)), self.sent_sim.predict(self.orig_sentence, " ".join(sentence))

    def mutate(self, sentence: SENTENCE_T) -> SENTENCE_T:
        return self.modifier.modify(sentence)

    def select(self, sentencepool: list[SENTENCE_INFO_T]):
        result: list[SENTENCE_INFO_T] = []
        best_similarity = max([x[2] for x in result])

        def scoring(t: float, s: float, bs: float):
            return (1 - t) * (s / bs)

        return sorted(result, key=lambda x: scoring(x[1], x[2], best_similarity), reverse=True)[:self.MAX_POOL_SIZE]

    def create_generation(
            self, sentence: SENTENCE_T,
            generation_num: int,
            children_limit: int = 10
    ) -> list[SENTENCE_INFO_T]:
        result: list[SENTENCE_INFO_T] = []

        children_limit = max(self.MIN_NUM_CHILDREN, children_limit)
        for _ in range(children_limit):
            modified_sentence = self.mutate(sentence)
            toxic_score, similarity_score = self.fitness(modified_sentence)

            result.append((modified_sentence, toxic_score, similarity_score))
            self.db.add_record(" ".join(modified_sentence), toxic_score, similarity_score, generation_num)
            print("|", end="", flush=True)

        return result

    def start_search(self, sentence: str, num_generations: int = 25):
        self.orig_sentence = sentence
        print("start search")

        tokenized_sentence: SENTENCE_T = sent_tokenize(sentence)
        toxicity_initial, similarity_initial = self.fitness(tokenized_sentence)
        list_of_sentences: list[SENTENCE_INFO_T] = [(tokenized_sentence, toxicity_initial, similarity_initial)]
        for _ in range(num_generations):
            print(f"Generation {_}")

            result: list[tuple[list[str], float, float]] = deepcopy(list_of_sentences)
            for item in list_of_sentences:
                result.extend(self.create_generation(item[0], _))

            sentences_only = [x[0] for x in list_of_sentences]
            for _ in range(self.MIN_CROSSOVER):
                sample = crossover(sentences_only)
                t, s = self.fitness(sample)
                result.append((sample, t, s))

            print(f"min toxicity achieved: {min([x[1] for x in result])}")
            list_of_sentences = self.select(list_of_sentences)
