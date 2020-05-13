import itertools

from nltk.translate.bleu_score import sentence_bleu
from pycorenlp import StanfordCoreNLP


class Helper:
    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost:9000')

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

    @staticmethod
    def find_default(items, item, default=-1):
        if item in items:
            return items.index(item)
        return default
