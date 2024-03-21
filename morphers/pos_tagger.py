import time
from pprint import pprint

import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

from utils.stats import timing
from utils.util import get_freer_gpu


class POSTagger:
    def __init__(self):

        self.model_name = "QCRI/bert-base-multilingual-cased-pos-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name).to("cuda")

        self.pipeline = TokenClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=int(get_freer_gpu()[-1]))

    @timing(name="POST_GEN")
    def generate(self, sentence):
        result = []
        overall = self.pipeline(sentence)
        for item in overall:

            if item["entity"].startswith("N"):
                result.append(f"{item['word']}.n")
            elif item["entity"].startswith("V"):
                result.append(f"{item['word']}.v")
            elif item["entity"].startswith("JJ"):
                result.append(f"{item['word']}.a")

        return result, overall

