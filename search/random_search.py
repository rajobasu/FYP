from models.similarity import SentenceSimilarityModel
from models.toxicity import ToxicityModel
from morphers.fancy_morpher import Morpher
from storage.simple_storage import Storage


class RandomSearch:
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

    def start_search(self, sentence: str, search_limit: int = 100):
        self.orig_sentence = sentence
        print(self.orig_sentence)
        for _ in range(search_limit):
            modified_sentence = self.modifier.modify(sentence)
            toxic_score = self.toxic.predict(modified_sentence)
            similarity_score = self.sent_sim.predict(self.orig_sentence, modified_sentence)

            self.db.add_record(modified_sentence, toxic_score, similarity_score)
            sentence = modified_sentence
