import json
import os
from operator import itemgetter

from pycorenlp import StanfordCoreNLP
import nltk
from nltk.tree import *

from helper import Helper

DIR = os.path.dirname(os.path.realpath(__file__))
PARSE_CONSTANTS_JSON = DIR + '/parse_constants.json'

with open(PARSE_CONSTANTS_JSON) as f:
    CONSTANTS = json.load(f)
    POS_CATEGORIES = CONSTANTS['pos_categories']
    SUBORDINATING_CONJUNCTIONS = CONSTANTS['subordinating_conjunctions']
    TENSES = CONSTANTS['tenses']
    MAIN_VERB_FORMS = CONSTANTS['main_verb_forms']
    AUX_VERB_FORMS = CONSTANTS['aux_verb_forms']
    PREPOSITIONS = CONSTANTS['prepositions']

CONSTANTS_FORMAT = '#constant({},{}).'
VARIABLES_FORMAT = 'var_{}({}).'


class ParseCoreNLP:
    def __init__(self, text, print_results=False):
        self.text = text
        self.print_results = print_results

        self.lemmas = {}  # Mapping of word -> lemma for ASG rules
        self.constants = set()  # Pairs of (category, lemma) to create ILASP constants

        # Load English tokenizer, tagger, parser, NER and word vectors
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.helper = Helper()

    # Returns a pair ([context_specific_asg], [ilasp_constants]) or ([context_specific_asg], [ilasp_background_vars])
    # If by_sentence is True, the output will be a list of pairs (one list item per sentence)
    def parse_text(self, by_sentence=False, background_vars=False):
        self._text_to_tree()
        if self.print_results:
            self.tree.pretty_print()
        return self._format_results(by_sentence=by_sentence, background_vars=background_vars)

    def _text_to_tree(self):
        output = self.nlp.annotate(self.text, properties={
            'annotators': 'tokenize,pos,lemma,depparse,parse',
            'outputFormat': 'json'
        })
        self._map_words_to_lemmas(output)

        sentences = [Tree.fromstring(sentence) for sentence in [s['parse'] for s in output['sentences']]]
        self.tree = sentences[0]

        # Combine trees
        for sentence in sentences[1:]:
            self.tree.append(sentence[0])

    def _map_words_to_lemmas(self, core_nlp_json):
        for sentence in core_nlp_json['sentences']:
            for token in sentence['tokens']:
                word = token['originalText']
                lemma = token['lemma']
                if word not in self.lemmas.keys():
                    self.lemmas[word] = lemma

    def _tree_to_asg(self, tree):
        asg_leaves = []
        if isinstance(tree[0], nltk.Tree):  # non-leaf node
            for subtree in tree:
                asg_leaves.extend(self._tree_to_asg(subtree))
        else:
            tag = tree.label().lower()
            word = tree[0]
            if word in self.lemmas.keys() and tag in POS_CATEGORIES.keys():
                lemma = self.lemmas[word].lower().replace('-', '_')

                if tag in TENSES.keys():
                    tense = TENSES[tag]
                    predicates = f'verb({lemma},{tense}). '
                    if tense in MAIN_VERB_FORMS:
                        self.constants.add(('main_verb', lemma))
                        self.constants.add(('main_form', tense))
                    elif tense in AUX_VERB_FORMS:
                        self.constants.add(('aux_verb', lemma))
                        self.constants.add(('aux_form', tense))
                else:
                    category = POS_CATEGORIES[tag]
                    predicates = f'{category}({lemma}). '
                    self.constants.add((category, lemma))
                asg_leaves.append(f'{tag} -> "{word} " {{ {predicates}}}')
        return asg_leaves

    # Takes as argument a string format with placeholders (category, lemma)
    def _lemmas_to_format(self, lemma_format):
        if lemma_format == CONSTANTS_FORMAT:
            for preposition in PREPOSITIONS:
                if preposition in list(map(itemgetter(1), self.constants)):
                    self.constants.add(('prep', preposition))
        return [lemma_format.format(category, lemma) for category, lemma in self.constants]

    def _format_results(self, tree=None, by_sentence=False, background_vars=False):
        if by_sentence:
            results = []
            for sent_tree in self.tree:
                context_specific_asg, ilasp_part = self._format_results(tree=sent_tree, background_vars=background_vars)
                sentence = str(sent_tree.flatten())[2:-1].strip()  # Remove brackets and node label from flattened tree
                results.append((context_specific_asg, ilasp_part, sentence))
                self.constants = set()
            return results

        tree = tree or self.tree
        context_specific_asg = sorted(set(self._tree_to_asg(tree)))
        if background_vars:
            ilasp_part = sorted(self._lemmas_to_format(VARIABLES_FORMAT))
        else:
            ilasp_part = sorted(self._lemmas_to_format(CONSTANTS_FORMAT))
        return context_specific_asg, ilasp_part
