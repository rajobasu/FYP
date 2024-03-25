# https://github.com/Mchristos/lexsub?tab=readme-ov-file

from gensim.models import KeyedVectors

from constants import ENV_VARS
from morphers.lexsub_1.lexsub import LexSub
from utils.stats import timing


class LexicalSubstitutor:
    def __init__(self):
        self.word2vec_path = f"{ENV_VARS['GENSIM_DATA']}/GoogleNews-vectors-negative300.bin"
        self.vectors = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
        self.ls = LexSub(self.vectors, candidate_generator='lin')
        sentence = "She had a drink at the bar"
        target = "bar.n"
        self.ls.lex_sub(target, sentence)

    @timing(name="LST_GEN")
    def generate(self, sentence, word_POS, batch_size=5):
        try:
            return self.ls.lex_sub(word_POS, sentence, batch_size=batch_size)
        except Exception as e:
            print(f"{sentence} <> {word_POS}")
            raise e
