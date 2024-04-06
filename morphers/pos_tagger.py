from typing import Any

from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

from constants import FREE_CUDA_ID
from utils.stats import timing


def post_processing(overall: Any):
    # we do some cleaning here since bad word and such might get split up into multiple words and each might
    # possibly be tagged as different parts of speech. moreover we need to combine them again.
    overall_modified: list[list[Any]] = []
    result = []
    ctr = 0
    for item in overall:
        ent = item["entity"]
        word = item["word"]

        if word.startswith("##"):
            if overall_modified[-1][0][-1] != ent:
                overall_modified[-1][0].append(ent)
            overall_modified[-1][1].append(word)
            continue

        overall_modified.append([[ent], [word], ctr])
        ctr += 1

    # if a word is split into multiple parts with different parts of speech assigned to different parts of the word,
    # we simply skip the word from our consideration.
    overall_final: list[tuple[str, str, int]] = []

    for item in overall_modified:
        ent_l, word_l, index = item
        if len(ent_l) > 1:
            ent_l[0] = ""  ## hacky way

        final_word = "".join([x.replace("#", "") for x in word_l])
        overall_final.append((ent_l[0], final_word, index))

    for item in overall_final:
        if item[0].startswith("N"):
            result.append((f"{item[1]}.n", item[2]))
        elif item[0].startswith("V"):
            result.append((f"{item[1]}.v", item[2]))
        elif item[0].startswith("JJ"):
            result.append((f"{item[1]}.a", item[2]))

    return result, overall_final


class POSTagger:
    def __init__(self):

        self.model_name = "QCRI/bert-base-multilingual-cased-pos-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name).to("cuda")

        self.pipeline = TokenClassificationPipeline(model=self.model, tokenizer=self.tokenizer,
                                                    device=int(FREE_CUDA_ID[-1]))

    # @timing(name="POST_GEN")
    def generate(self, sentence: str):
        overall = self.pipeline(sentence)
        return post_processing(overall)

    @timing(name="POS_BATCH")
    def generate_batch(self, sentences: list[str]):
        return [post_processing(x) for x in self.pipeline(sentences, batch_size=16)]
