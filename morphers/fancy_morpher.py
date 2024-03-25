import random
from abc import ABC, abstractmethod
from pprint import pprint

from nltk import sent_tokenize
from transformers import pipeline

from dataparser import get_random_word, get_data
from morphers.paraphraser import Paraphraser
from morphers.synonym_substitutor import LexSubWrapper


class Morpher(ABC):
    @abstractmethod
    def modify(self, text):
        pass

    def modify_batch(self, texts, children_per_sentence):
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


def text_to_list(text):
    if type(text) is list:
        return text
    return sent_tokenize(text)


def generate_list_using(gen_func, texts: list[list[str]], num_children):
    sentence_per_text = len(texts[0])
    for text in texts:
        if len(text) != sentence_per_text:
            print("SENTENCES NOT THE SAME LENGTH")
            print(texts)
            raise Exception()
    all_texts = []
    for text in texts:
        all_texts.extend(text)  # since text is also a list of sentences

    outputs = gen_func(all_texts, num_children)
    result = []

    for text_id in range(len(texts)):
        for child_id in range(num_children):
            sent = []
            for sentence_id in range(sentence_per_text):
                sent.append(outputs[text_id * num_children * sentence_per_text + child_id + sentence_id * num_children])
            result.append(sent)

    return result


class SynonymParaphraserMorper(Morpher):
    def __init__(self):
        self.lws = LexSubWrapper()
        self.ppr = Paraphraser()

    def modify_list(self, sent_list):
        if random.random() < 0.3:
            sent_list = [self.ppr.generate(x) for x in sent_list]

        sent_list = [self.lws.generate(x) for x in sent_list]
        return sent_list

    def modify(self, text):
        if type(text) is list:
            return self.modify_list(text)

        sent_list = sent_tokenize(text)
        return " ".join(self.modify_list(sent_list))

    ## assumption here is that exery text:list in texts has the same length
    def modify_batch(self, texts, children_per_sentence, ratio=0.5):
        # we just do replacement for half, and paraphrase for half

        texts = [text_to_list(text) for text in texts]
        morph_changes = int(ratio * children_per_sentence)
        lex_changes = int(children_per_sentence - morph_changes)

        results = []
        results.extend(generate_list_using(
            gen_func=self.ppr.generate_batch,
            texts=texts,
            num_children=morph_changes
        ))

        results.extend(
            generate_list_using(
                gen_func=self.lws.batch_generate,
                texts=texts,
                num_children=lex_changes
            )
        )

        return results
