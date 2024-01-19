import random

from transformers import pipeline
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataparser import get_random_word


class Morpher(ABC):
    @abstractmethod
    def modify(self, text) -> str:
        pass


class QualityControlPipeline(Morpher):
    def __init__(self, type="sentences"):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}')
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def modify2(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
            f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']
        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control = {name: max(min(val, self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in
                   zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)[0]['generated_text']

    def modify(self, text):
        self.modify2(text, 0.3, 0.4, 0.5)


"""
This morpher replaces words entirely at random from a preselected word list. 
"""


class RandomMorpher(Morpher):
    def __init__(self):
        pass


    def add_word(self, sentence):
        """Append a word to the end of the sentence"""
        sentence += " " + get_random_word()
        return sentence

    def word_replacement(self, sentence):
        """Take a random word in the sentence and modify it"""
        words = sentence.split(" ")
        words[random.randint(0, len(words) - 1)] = get_random_word()
        return " ".join(words)

    def modify(self, sentence):
        randval = random.randint(0, 100)
        if randval < 30:
            return self.add_word(sentence)

        return self.word_replacement(sentence)