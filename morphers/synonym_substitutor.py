from pprint import pprint
import random
import re
from morphers.lexsub_1.lexsub_test import LexicalSubstitutor
from morphers.pos_tagger import POSTagger
from utils.stats import timing
from utils.util import debug_print


class LexSubWrapper:
    def __init__(self):
        self.postagger = POSTagger()
        self.subs = LexicalSubstitutor()

    @timing(name="LXSW_GEN")
    def generate(self, sentence, n: int = 2):
        modified_sentence = re.sub("\w*\*+\w*", "", sentence)
        parts, overall = self.postagger.generate(modified_sentence)

        if not parts:
            return sentence

        # choose how many words to replace.
        n = min(n, len(parts))
        if n == -1:
            n = random.randint(1, len(parts))
        # randomly select n words to replace. if there are not replacement words found, we just skip those words.
        # in effect this means that not all words might be replaced.
        random.shuffle(parts)
        for index in range(n):
            target, sent_index = parts[index]
            replacements = self.subs.generate(sentence, target)
            if not replacements:
                continue

            e, _, i = overall[sent_index]
            overall[sent_index] = e, random.choice(replacements), i

        output = " ".join([x[1] for x in overall])
        # debug_print(f"CHANGED: [[{sentence}]] > [[{output}]]")
        return output

    # def batch_generate(self):
