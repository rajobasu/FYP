from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer, util

from constants import FREE_CUDA_ID


class SentenceSimilarityModel(ABC):
    @abstractmethod
    def predict(self, sentence1: str, sentence2: str) -> float:
        pass


class MiniLM(SentenceSimilarityModel):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2').to(device=FREE_CUDA_ID)

    def predict(self, sentence1, sentence2) -> float:
        embeddings1 = self.model.encode([sentence1], convert_to_tensor=True).to(device=FREE_CUDA_ID)
        embeddings2 = self.model.encode([sentence2], convert_to_tensor=True).to(device=FREE_CUDA_ID)

        return util.cos_sim(embeddings1, embeddings2)[0][0].item()
