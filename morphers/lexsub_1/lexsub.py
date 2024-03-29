""" Different models for performing lexical substitution task
    (find substitution for a word in context)"""
# https://github.com/Mchristos/lexsub/blob/master/lexsub.py
from gensim.models import KeyedVectors
from nltk.corpus import lin_thesaurus as lin
from nltk.corpus import wordnet as wn

from morphers.lexsub_1 import tools
from morphers.lexsub_1.tools import process_candidates
# https://github.com/Mchristos/lexsub/blob/master/lexsub.py
from gensim.models import KeyedVectors
from nltk.corpus import lin_thesaurus as lin
from nltk.corpus import wordnet as wn

from morphers.lexsub_1 import tools
from morphers.lexsub_1.tools import process_candidates

WORD2VEC_PATH = "~/GoogleNews-vectors-negative300-SLIM.bin"


def lin_synonyms(word, pos):
    fileid = 'sim%s.lsp' % pos.upper()
    thes_entry = lin.scored_synonyms(word, fileid=fileid)
    thes_entry = sorted(thes_entry, key=(lambda x: x[1]), reverse=True)
    # return words ordered by score
    return [syn for syn, score in thes_entry]


def wordnet_synonyms(word, pos):
    if pos == 'n':
        synset = wn.synsets(word, wn.NOUN)
        # return synonym lemmas in no particular order
        return [lemma.name() for s in synset for lemma in s.lemmas()]
    else:
        raise ValueError("unsupported part of speech: %r" % pos)


class LexSub(object):
    "Find word substitutions for a word in context using word2vec skip-gram embedding"

    def __init__(self, word_vectors=None, candidate_generator='word2vec', n_candidates=50):
        """
        n_substitutes = number of lexical substitutes to generate
        candidate_generator = word2vec, lin, wordnet
        """
        # supported POS values
        self.poses = ['n', 'a', 'v']
        # number of generated candidates for substitution
        self.n_candidates = n_candidates
        if word_vectors is None:
            self.word_vectors = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        else:
            self.word_vectors = word_vectors
        if candidate_generator in ['word2vec', 'lin', 'wordnet']:
            self.candidate_generator = candidate_generator
        else:
            raise ValueError("Invalid value candidate_generator: %r" % candidate_generator)

    def get_candidates(self, word, POS):
        if self.candidate_generator == 'word2vec':
            words_scores = self.word_vectors.most_similar(positive=[word])
            result = [word for word, score in words_scores]
        if self.candidate_generator == 'lin':
            result = lin_synonyms(word, POS)
        if self.candidate_generator == 'wordnet':
            result = wordnet_synonyms(word, POS)
        # words to lower case, replace underscore, remove duplicates,
        # remove target word and stop words, clip length
        result = process_candidates(result, word)[:self.n_candidates]
        assert (len(result) <= self.n_candidates)
        return result

    def get_substitutability(self, t, s, C):
        """ get substitutability of substitution s for target t in context C
        t = target word
        s = candidate substitution
        C = list of context words
        """
        # 1. target score: how similar is it to the target word?
        tscore = self.word_vectors.similarity(t, s)
        # 2. context score: how similar is it to the context words?
        cscores = [self.word_vectors.similarity(s, c) for c in C]
        cscore = sum(cscores)
        return (tscore + cscore) / (len(C) + 1)

    def lex_sub(self, word_POS, sentence, batch_size=5):
        """ Get appropriate substitution for a word given context words

        word_POS = word with part of speech in form word.POS e.g. dog.n
        context_words = list of words in context
        """

        try:
            w, _, POS = word_POS.partition('.')
            # generate candidate substitutions
            candidates = self.get_candidates(w, POS)
            if sentence is None:
                return candidates[:batch_size]
            else:
                context_words = tools.get_words(sentence)
                # filter context words: exist in the word2vec vocab, not stop words
                context_words = list(filter(lambda c: c in self.word_vectors.key_to_index
                                                      and c not in tools.stopwords,
                                            context_words))
                cand_scores = [
                    self.get_substitutability(w, s, context_words) if s in self.word_vectors.key_to_index else 0
                    for s
                    in candidates]
                assert (len(cand_scores) == len(candidates))
                sorted_candidates = sorted(zip(candidates, cand_scores), key=lambda x: x[1], reverse=True)
                return [sub for sub, score in sorted_candidates][:batch_size]
        except KeyError:
            print(f"Key {w} was not present in word2vec?")
            return []
