from pprint import pprint
from random import randrange
import random

from models.similarity import SentenceSimilarityModel
from models.toxicity import ToxicityModel
from morphers.fancy_morpher import Morpher
from storage.simple_storage import Storage


def get_random_item(list_item):
    return list_item[randrange(0, len(list_item))]


class IterativeSearch:
    def __init__(self,
                 toxicity_rater: ToxicityModel,
                 modifier: Morpher,
                 similarity_rater: SentenceSimilarityModel,
                 db: Storage):
        self.toxic = toxicity_rater
        self.sent_sim = similarity_rater
        self.modifier = modifier
        self.db = db
        self.orig_sentence = ""

    def start_search(self, sentence: str, search_limit: int = 3):
        self.orig_sentence = sentence
        pprint("start search")
        for _ in range(search_limit):
            pprint(f"Searching item {_}")
            modified_sentence = self.modifier.modify(sentence)
            toxic_score = self.toxic.predict(modified_sentence)
            similarity_score = self.sent_sim.predict(self.orig_sentence, modified_sentence)

            self.db.add_record(modified_sentence, toxic_score, similarity_score)
            sentence = modified_sentence


class PopulationSearch:
    def __init__(self,
                 toxicity_rater: ToxicityModel,
                 modifier: Morpher,
                 similarity_rater: SentenceSimilarityModel,
                 db: Storage):

        self.toxic = toxicity_rater
        self.sent_sim = similarity_rater
        self.modifier = modifier
        self.db = db
        self.orig_sentence = ""

    def is_bad(self, sent, tox, sim, list_of_sentences):
        res = []
        for a, b, c in list_of_sentences:
            if b < tox and c > sim:
                return list_of_sentences
            elif b > tox and c < sim:
                pass
            else:
                res.append((a, b, c))
        res.append((sent, tox, sim))
        return res

    def start_search(self, sentence: str, search_limit: int = 1000):
        self.orig_sentence = sentence
        pprint("start search")
        list_of_sentences = [(sentence, self.toxic.predict(sentence), 1)]
        for _ in range(search_limit):
            pprint(f"Searching item {_}")
            modified_sentence = self.modifier.modify(get_random_item(list_of_sentences)[0])
            toxic_score = self.toxic.predict(modified_sentence)
            similarity_score = self.sent_sim.predict(self.orig_sentence, modified_sentence)

            self.db.add_record(modified_sentence, toxic_score, similarity_score)

            list_of_sentences = self.is_bad(modified_sentence, toxic_score, similarity_score, list_of_sentences)
            print(toxic_score, similarity_score)

        pprint(sorted([(b, c) for a, b, c in list_of_sentences]))


class PopulationSearchWithLooseFrontier:
    def __init__(self,
                 toxicity_rater: ToxicityModel,
                 modifier: Morpher,
                 similarity_rater: SentenceSimilarityModel,
                 db: Storage):

        self.toxic = toxicity_rater
        self.sent_sim = similarity_rater
        self.modifier = modifier
        self.db = db
        self.orig_sentence = ""

    def is_bad(self, sent, tox, sim, list_of_sentences):
        res = []
        for a, b, c in list_of_sentences:
            randval = random.randint(0, 100)
            if randval < 10:
                res.append((a, b, c))
                continue

            if b < tox and c > sim:
                return list_of_sentences
            elif b > tox and c < sim:
                pass
            else:
                res.append((a, b, c))
        res.append((sent, tox, sim))
        return res

    def start_search(self, sentence: str, search_limit: int = 1000):
        self.orig_sentence = sentence
        pprint("start search")
        list_of_sentences = [(sentence, self.toxic.predict(sentence), 1)]
        for _ in range(search_limit):
            pprint(f"Searching item {_}")
            modified_sentence = self.modifier.modify(get_random_item(list_of_sentences)[0])
            toxic_score = self.toxic.predict(modified_sentence)
            similarity_score = self.sent_sim.predict(self.orig_sentence, modified_sentence)

            self.db.add_record(modified_sentence, toxic_score, similarity_score)

            list_of_sentences = self.is_bad(modified_sentence, toxic_score, similarity_score, list_of_sentences)
            print(toxic_score, similarity_score)

        pprint(sorted([(b, c) for a, b, c in list_of_sentences]))


def scoring(toxicity: float, similarity: float, scoring_func_type: int = 1):
    if scoring_func_type == 1:
        return (1 - toxicity) * similarity


class PopulationBasedIterativeSearch:
    def __init__(self,
                 toxicity_rater: ToxicityModel,
                 modifier: Morpher,
                 similarity_rater: SentenceSimilarityModel,
                 db: Storage):

        self.toxic = toxicity_rater
        self.sent_sim = similarity_rater
        self.modifier = modifier
        self.db = db
        self.orig_sentence = ""

    def create_generation(self, sentence: str, children_limit: int = 20) -> list[tuple[str, float, float]]:
        result: list[tuple[str, float, float]] = []
        for _ in range(children_limit):
            modified_sentence = self.modifier.modify(sentence)
            toxic_score = self.toxic.predict(modified_sentence)
            similarity_score = self.sent_sim.predict(self.orig_sentence, modified_sentence)
            result.append((modified_sentence, toxic_score, similarity_score))

        return result

    def start_search(self, sentence: str, num_generations: int = 50):
        self.orig_sentence = sentence
        pprint("start search")
        list_of_sentences = [sentence]
        for _ in range(num_generations):
            pprint(f"Generation {_}")
            result = []
            for sentence in list_of_sentences:
                result.extend(self.create_generation(sentence))

            for modified_sentence, toxic_score, similarity_score in result:
                print(toxic_score, similarity_score)
                self.db.add_record(modified_sentence, toxic_score, similarity_score)

            scored_results = [(scoring(tox, sim), sent) for sent, tox, sim in result]
            scored_results.sort(key=lambda x: x[0], reverse=True)
            list_of_sentences = [sent for score, sent in scored_results[:10]]
