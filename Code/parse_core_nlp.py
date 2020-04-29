import json
import os

from pycorenlp import StanfordCoreNLP
import nltk
from nltk.tree import *

from helper import Helper

DIR = os.path.dirname(os.path.realpath(__file__))
PARSE_CONSTANTS_JSON = DIR + '/parse_constants.json'

with open(PARSE_CONSTANTS_JSON) as f:
    constants = json.load(f)
    PUNCTUATION = constants['punctuation']
    POS_CATEGORIES = constants['pos_categories']
    TENSES = constants['tenses']

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
    def parse_text(self, background_variables=False):
        self._text_to_tree()
        self._remove_punctuation_nodes(self.tree)
        return self._format_results(background_variables)

    def _text_to_tree(self):
        output = self.nlp.annotate(self.text, properties={
            'annotators': 'tokenize,pos,lemma,depparse,parse',
            'outputFormat': 'json'
        })
        self._map_words_to_lemmas(output)

        sentences = [Tree.fromstring(sentence) for sentence in [s['parse'] for s in output['sentences']]]
        self.tree = sentences[0]

        # Combine trees
        itersentences = iter(sentences)
        next(itersentences)
        [self.tree.insert(len(self.tree[0]), sentence[0]) for sentence in itersentences]

    def _map_words_to_lemmas(self, core_nlp_json):
        for sentence in core_nlp_json['sentences']:
            for token in sentence['tokens']:
                word = token['originalText']
                lemma = token['lemma']
                if word in self.lemmas.keys():
                    assert self.lemmas[word] == lemma, "Multiple lemmas {} and {} for word {}".format(self.lemmas[word], lemma, word)
                else:
                    self.lemmas[word] = lemma

    # TODO improve efficiency
    def _remove_punctuation_nodes(self, tree):
        i = 0
        tree_size = len(tree)
        while i < tree_size:
            node = tree[i]
            if not isinstance(node, nltk.Tree):
                return
            if node.label() in PUNCTUATION:
                del tree[i]
                tree_size -= 1
            else:
                ParseCoreNLP._remove_punctuation_nodes(self, node)
                i += 1

    def _tree_to_asg(self, tree, asg_leaves=[]):
        if isinstance(tree[0], nltk.Tree):  # non-leaf node
            [self._tree_to_asg(subtree, asg_leaves) for subtree in tree]
        else:
            tag = tree.label().lower()
            word = tree[0]
            if word in self.lemmas.keys() and tag in POS_CATEGORIES.keys():
                category = POS_CATEGORIES[tag]
                lemma = self.lemmas[word].lower()
                predicates = self.helper.get_base_predicates(tag, lemma)

                self.constants.add((category, lemma))
                if tag in TENSES.keys():
                    predicates = f'verb({lemma},{TENSES[tag]}). '
                    self.constants.add(('verb_form', TENSES[tag]))
                else:
                    predicates = f'{category}({lemma}). '
                asg_leaves.append(f'{tag} -> "{word} " {{{predicates}}}')
        return asg_leaves

    # Takes as argument a string format with placeholders (category, lemma)
    def _lemmas_to_format(self, lemma_format):
        return [lemma_format.format(category, lemma) for category, lemma in self.constants]

    def _format_results(self, background_variables=False):
        context_specific_asg = sorted(set(self._tree_to_asg(self.tree)))
        if background_variables:
            ilasp_part = sorted(self._lemmas_to_format(VARIABLES_FORMAT))
        else:
            ilasp_part = sorted(self._lemmas_to_format(CONSTANTS_FORMAT))

        if self.print_results:
            self.tree.pretty_print()
            print(context_specific_asg)
            print(ilasp_part)
        return context_specific_asg, ilasp_part
