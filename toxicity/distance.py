import math

import numpy as np

import utils.util
from morphers.fancy_morpher import RandomMorpher
from toxicity.toxicity import ToxicityEvaluator, BooleanToxicityEvaluatorWrapper
from utils.stats import timing

random_morpher = RandomMorpher()

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

        self.mid = 0

    def get_setup_question(self):

        return [self.generated_sentences[0], self.generated_sentences[-1]]

    def answer_setup_question(self, val1, val2):
        if val1 == 0:
            self.lo = self.hi = 0
        elif val2 == 1:
            self.lo = self.hi = self.limit

    def get_next_question(self):
        if self.lo >= self.hi - 1:  # in effect if they have a gap of 2 or less
            return None

        self.mid = int(math.ceil((self.lo + self.hi) / 2))  # mid
        return self.generated_sentences[self.mid]

    def update_with_answer(self, answer):
        if answer is None:
            return

        print(f"BEFORE: [{self.lo}, {self.hi}]", end="")
        if answer == 1:
            self.lo = self.mid
        elif answer == 0:
            self.hi = self.mid

        print(f" AFTER: [{self.lo}, {self.hi}]")

    def get_answer(self):
        return self.lo / self.limit


def batch_distance(*, sentences: list[str], evaluator: BooleanToxicityEvaluatorWrapper, limit: int) -> list[float]:
    searchers = [BinarySearcher(toxicity_rater=evaluator, limit=limit, sentence=sentence) for sentence in sentences]
    setup_questions = []
    for searcher in searchers:
        setup_questions.extend(searcher.get_setup_question())
    setup_answers = evaluator.predict_batch(setup_questions)
    print("\n"*5)
    print(setup_questions)
    print(f"SETUP {setup_answers}")
    print("\n"*5)
    for searcher, ans in zip(searchers, utils.util.split_batch(setup_answers, 2)):
        searcher.answer_setup_question(ans[0], ans[1])

    ctr = 0
    while True:
        print("NEW ITERATION OF BINARY_SEARCH")
        ctr += 1
        if ctr >= 20:
            print("CATASTROPHIC FAILURE")
            exit(0)

        ids = []
        sentences = []
        for i, searcher in enumerate(searchers):
            sent = searcher.get_next_question()
            if sent is not None:
                ids.append(i)
                sentences.append(sent)

        print(f"SEARCHERS PARTICIPATED: {len(ids)} ")
        if not ids:
            break

        results = evaluator.predict_batch(sentences)
        for i, res in zip(ids, results):
            searchers[i].update_with_answer(res)

    return [searcher.get_answer() for searcher in searchers]
