import random

from morphers.lexsub_1.lexsub_test import LexicalSubstitutor
from morphers.pos_tagger import POSTagger


class LexSubWrapper:
    def __init__(self):
        self.postagger = POSTagger()
        self.subs = LexicalSubstitutor()

    def generate(self, sentence, n: int = -1):
        parts, overall = self.postagger.generate(sentence)
        new_parts: list[tuple[str, int]] = []
        for val, ind in parts:
            if val.startswith("##"):
                new_parts.pop()
            else:
                new_parts.append((val, ind))
        parts = new_parts

        if not parts:
            return sentence

        n = min(n, len(parts))
        if n == -1:
            n = random.randint(1, len(parts))

        random.shuffle(parts)
        print(f"n : {n}")
        for index in range(n):
            target, sent_index = parts[index]
            replacements = self.subs.generate(sentence, target)
            print(f"{target} : {replacements}")
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
