import random
import re

from morphers.lexsub_1.lexsub_test import LexicalSubstitutor
from morphers.pos_tagger import POSTagger
from utils.stats import timing


class LexSubWrapper:
    def __init__(self):
        self.postagger = POSTagger()
        self.subs = LexicalSubstitutor()

    def cleanup(self, sentences):
        return [re.sub("\w*\*+\w*", "", sentence) for sentence in sentences]

    @timing(name="LXSW_GEN")
    def generate(self, sentence, n: int = 2):
        modified_sentence = self.cleanup([sentence])[0]
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

    @timing(name="LXSW_Batch_Gen")
    def batch_generate(self, sentences, children_per_sentence):
        # each new generation is one replacement only.
        sentences = self.cleanup(sentences)
        pos_tagged_items = self.postagger.generate_batch(sentences)
        result = []

        for (parts, overall), sentence in zip(pos_tagged_items, sentences):

            # we will now make a list of all the possible substitutions
            replacements_left = children_per_sentence

            random.shuffle(parts)
            if not parts:
                result.append(sentence)

            for target, sent_index in parts:
                if replacements_left <= 0:
                    break

                r = self.subs.generate(sentence, target, replacements_left)
                if not r:
                    continue

                e, _, i = overall[sent_index]
                for repl in r:
                    overall[sent_index] = e, repl, i
                    result.append(" ".join([x[1] for x in overall]))

                overall[sent_index] = e, _, i
                replacements_left -= len(r)

        return result
