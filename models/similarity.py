from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, util


class SentenceSimilarityModel(ABC):
    @abstractmethod
    def predict(self, sentence1: str, sentence2: str) -> float:
        pass


class MiniLM(SentenceSimilarityModel):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def predict(self, sentence1, sentence2) -> float:
        embeddings1 = self.model.encode([sentence1], convert_to_tensor=True)
        embeddings2 = self.model.encode([sentence2], convert_to_tensor=True)

        return util.cos_sim(embeddings1, embeddings2)[0][0].item()
