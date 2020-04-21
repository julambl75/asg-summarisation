import itertools

import inflect
from nltk.translate.bleu_score import sentence_bleu
from pycorenlp import StanfordCoreNLP


class Helper:
    def __init__(self):
        self.p = inflect.engine()
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    # Note: this does not work in general for nouns; for example p.plural_noun('cars') returns 'cars'
    def is_plural_pronoun(self, word):
        return self.p.plural_noun(word) == word

    def is_plural_verb(self, word):
        return self.p.plural_verb(word) == word

    # Add necessary predicates to ASG leaf nodes for edge cases
    def get_base_predicates(self, tag, lemma):
        # TODO fix edge case
        if tag == 'prp' and lemma != 'you':
            is_plural = self.is_plural_pronoun(lemma)
            return self._form_gram_num_predicate(lemma, is_plural)
        # elif tag == 'vbd':
        #     is_plural = self.is_plural_verb(lemma)
        #     return self._form_gram_num_predicate(lemma, is_plural)
        elif tag == 'vbg':
            return ' vb_obj_match(no_obj). '
        return ' '

    @staticmethod
    def _form_gram_num_predicate(word, is_plural):
        gram_num = 'plural' if is_plural else 'singular'
        return " {}({}). ".format(gram_num, word)

    # Returns a list of lists of word POS tags, one for each sentence,
    #   or simply a list of each word's POS tag is ignore_sentence is True
    def tokenize_text(self, text, ignore_sentence=False):
        output = self.nlp.annotate(text, properties={
            'annotators': 'pos',
            'outputFormat': 'json'
        })
        tokenized = [[(word['word'], word['pos']) for word in sentence['tokens']] for sentence in output['sentences']]
        if ignore_sentence:
            return list(itertools.chain.from_iterable(tokenized))
        return tokenized

    # BLEU score evaluates the quality of machine-translated text
    @staticmethod
    def bleu_score(summary, reference):
        return sentence_bleu([reference], summary)

    @staticmethod
    def count_sentences(text):
        return len(list(filter(lambda s: len(s) > 0, text.strip().split('.'))))
