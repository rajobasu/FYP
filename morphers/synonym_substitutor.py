import random
import re
from morphers.lexsub_1.lexsub_test import LexicalSubstitutor
from morphers.pos_tagger import POSTagger
from utils.stats import timing


class LexSubWrapper:
    def __init__(self):
        self.postagger = POSTagger()
        self.subs = LexicalSubstitutor()

    @timing(name="LXSW_GEN")
    def generate(self, sentence, n: int = 2):
        modified_sentence = re.sub("\w*\*+\w*", "", sentence)
        orig_parts, overall = self.postagger.generate(modified_sentence)
        fixed_parts: list[str] = []
        for val in orig_parts:
            if val.startswith("##"):
                fixed_parts[-1] = fixed_parts[-1][:-2] + val[2:]
            else:
                fixed_parts.append(val)

        parts = list(enumerate(fixed_parts))

        if not parts:
            return sentence


        n = min(n, len(parts))
        if n == -1:
            n = random.randint(1, len(parts))
        random.shuffle(parts)
        for index in range(n):
            sent_index, target = parts[index]
            replacements = self.subs.generate(sentence, target)
            if not replacements:
                continue
            overall[sent_index - 1]['word'] = random.choice(replacements)

        result: list[str] = []
        for item in overall:
            if item['word'].startswith("##"):
                result[-1] = result[-1] + item['word'][2:]
            else:
                result.append(item['word'])

        return " ".join(result)


    # def batch_generate(self):