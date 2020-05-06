import argparse
import itertools
import math
import pprint as pp
import re
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

from ordered_set import OrderedSet

from helper import Helper
from parse_concept_net import ParseConceptNet

warnings.filterwarnings("ignore")

SUBSTITUTIONS = {'an': 'a'}

VERB_POS = 'VB'
PROPER_POS = 'NNP'
IGNORE_POS = ['DT', '.']
SAME_WORD_POS = ['NN', 'NNS', 'NNP', 'NNPS']
COMPLEX_CLAUSE_AUX_VERB_POS = 'VBN'
COMPLEX_CLAUSE_SPLIT_VERBS = [('was', 'VBD'), ('is', 'VBZ')]
COMPLEX_CLAUSE_SUBSTITUTIONS = {'a': 'The'}
EOS_TOKENIZED = ('.', '.')

SAME_WORD_SIMILARITY = 5
WEIGHT_SCALE = 10
MIN_SYNONYM_SIMILARITY = 2
SENT_IMPORTANCE_SQRT = 10
MIN_NUM_SENT_FOR_PRUNE = 3


class Preprocessor:
    def __init__(self, story, print_results=True, proper_nouns=False):
        self.story = story.strip().replace('\n', ' ')
        self.print_results = print_results
        self.proper_nouns = proper_nouns

        self.helper = Helper()
        self.pcn = ParseConceptNet(False)

    def preprocess(self):
        if self.print_results:
            pp.pprint(self.story)
        self._substitute_determiners()

        tokenized = self.helper.tokenize_text(self.story)
        tokenized = self._expand_complex_clauses(tokenized)
        tokenized = self._split_conjunctive_clauses(tokenized)
        self._replace_story_new_tokenized(tokenized)

        if self.print_results:
            print('\nGenerating POS tags...')
            pp.pprint(tokenized)
        similar_words, similar_sentences, vocabulary = self._process_similarity(tokenized)

        if self.print_results:
            print('\nGenerating word and sentence adjacency matrices...')
        word_adjacency_mat = self._links_dict_to_adjacency_mat(similar_words, vocabulary)
        sentence_adjacency_mat = self._links_dict_to_adjacency_mat(similar_sentences, range(len(tokenized)))

        if self.print_results:
            print('\nPlotting similarity between words and sentences...')
            labels = {i: vocabulary[i] for i in range(len(vocabulary))}
            self._plot_text_relationship_map(word_adjacency_mat, 'Word Relationship Map', labels)
            self._plot_text_relationship_map(sentence_adjacency_mat, 'Sentence Relationship Map')

        ordered_sentences = self._order_sentences_by_importance(sentence_adjacency_mat, self.story)
        if self.print_results:
            print('\nOrdering sentences by importance...')
            pp.pprint(ordered_sentences)

        synonyms = self._get_synonyms(similar_words)
        if self.print_results:
            print('\nBuilding synonyms...')
            pp.pprint(synonyms)

        shortest_word_map = self._get_shortest_word_map(synonyms)
        if self.print_results:
            print('\nGetting shortest words for each set of synonyms...')
            pp.pprint(shortest_word_map)

        homogenized_story = self._homogenize_text(self.story, shortest_word_map, ordered_sentences)
        if self.print_results:
            print('\nHomogenizing story using synonyms...')
            pp.pprint(homogenized_story)

        if self.proper_nouns:
            proper_nouns = {token[0] for token in itertools.chain(*tokenized) if token[1].startswith(PROPER_POS)}
            return homogenized_story, proper_nouns
        return homogenized_story

    def _substitute_determiners(self):
        for key, value in SUBSTITUTIONS.items():
            self.story = re.sub(r"\b{}\b".format(key), value, self.story)

    # TODO
    def _replace_punctuation(self):
        print('TODO')
        # ?–— -> X
        # !,; -> .
        pass

    # Ex: There was a boy named Peter.
    #  -> There was a boy. The boy was named Peter..
    def _expand_complex_clauses(self, tokenized):
        i = 0
        while i < len(tokenized):
            sentence = tokenized[i]
            pos_tags = list(map(itemgetter(1), sentence))
            if COMPLEX_CLAUSE_AUX_VERB_POS in pos_tags:
                aux_clause_idx = pos_tags.index(COMPLEX_CLAUSE_AUX_VERB_POS)
            else:
                aux_clause_idx = -1

            # If there is an auxiliary clause (starting with a VBN that does not follow a verb)
            if aux_clause_idx > 0 and not pos_tags[aux_clause_idx-1].startswith(VERB_POS):
                main_clause = sentence[:aux_clause_idx]
                aux_clause_obj = sentence[aux_clause_idx:]

                # Prepend to the auxiliary clause everything in the main clause after its last verb
                #   i.e., use the main clause's object as the subject of the auxiliary clause
                pos_tags_main = pos_tags[:aux_clause_idx]
                main_clause_verbs = list(filter(lambda t: t[1].startswith(VERB_POS), main_clause))
                main_clause_obj_idx = len(main_clause) - pos_tags_main[::-1].index(main_clause_verbs[-1][1])
                main_clause_obj = main_clause[main_clause_obj_idx:]

                # Change 'a' to 'the' at start of subject of the auxiliary sentence
                first_aux_word, first_aux_pos = main_clause_obj[0]
                if first_aux_word in COMPLEX_CLAUSE_SUBSTITUTIONS.keys():
                    first_aux_word = COMPLEX_CLAUSE_SUBSTITUTIONS[first_aux_word]
                    main_clause_obj[0] = (first_aux_word, first_aux_pos)
                aux_clause = main_clause_obj + main_clause_verbs + aux_clause_obj

                # Replace tokenized sentence with two tokenized sentences and check auxiliary clause next
                tokenized[i] = main_clause + [EOS_TOKENIZED]
                tokenized.insert(i+1, aux_clause)
            i += 1
        return tokenized

    def _split_conjunctive_clauses(self, tokenized):
        print('TODO')
        return tokenized

    def _replace_story_new_tokenized(self, tokenized):
        tokenized_words = list(map(itemgetter(0), itertools.chain.from_iterable(tokenized)))
        self.story = ' '.join(tokenized_words).replace(' .', '.')

    def _process_similarity(self, tokenized):
        similar_words = defaultdict(lambda: defaultdict(lambda: 0))
        similar_sentences = defaultdict(lambda: defaultdict(lambda: 0))
        vocabulary = OrderedSet()

        for i, sentence in enumerate(tokenized):
            for word, pos in sentence:
                if pos not in IGNORE_POS:
                    for j, other_sentence in enumerate(tokenized):
                        if i != j:
                            for other_word, other_pos in other_sentence:
                                word = word.lower()
                                other_word = other_word.lower()

                                if pos == other_pos:
                                    if word == other_word:
                                        similarity = SAME_WORD_SIMILARITY if pos in SAME_WORD_POS else 0
                                    else:
                                        similarity = self.pcn.compare_words(word, other_word)
                                    if similarity > 0:
                                        vocabulary.add(word)
                                        vocabulary.add(other_word)
                                        similar_words[word][other_word] = similarity
                                        similar_sentences[i][j] += similarity
        return similar_words, similar_sentences, vocabulary

    @staticmethod
    def _pp_print_links_dict(links_dict):
        pp.pprint({i: dict(links) for i, links in links_dict.items()})

    @staticmethod
    def _links_dict_to_adjacency_mat(links_dict, keys):
        adjacency_mat = []
        for key in keys:
            node_weights = [int(WEIGHT_SCALE * links_dict[key][other_key]) for other_key in keys]
            adjacency_mat.append(node_weights)
        return adjacency_mat

    @staticmethod
    def _plot_text_relationship_map(adjacency_mat, title, labels=None):
        if len(adjacency_mat) == 0:
            return
        graph = nx.from_numpy_matrix(np.matrix(adjacency_mat), create_using=nx.DiGraph)
        layout = nx.circular_layout(graph)
        nx.draw(graph, layout)
        nx.draw_networkx_labels(graph, pos=layout, labels=labels)
        nx.draw_networkx_edge_labels(graph, pos=layout)
        plt.title(title)
        plt.show()

    @staticmethod
    def _order_sentences_by_importance(adjacency_mat, story):
        sentences = [sentence.strip() for sentence in story.split('.')]
        normalization_factors = [math.ceil(len(sentence) ** 1. / SENT_IMPORTANCE_SQRT) for sentence in sentences if
                                 len(sentence) > 0]
        weights_sum = np.array(list(map(sum, adjacency_mat)))
        normalized_weights = np.around(weights_sum / normalization_factors, decimals=1)
        importance_ordering = np.argsort(-normalized_weights)
        ordered_sentences = [sentences[i] + '.' for i in importance_ordering]
        return list(zip(ordered_sentences, sorted(normalized_weights, reverse=True)))

    @staticmethod
    def _get_synonyms(similar_words):
        synonyms = defaultdict(set)
        word_in_synonyms = lambda w: w in set.union(*synonyms.values())

        for word, links in similar_words.items():
            for other_word, similarity in links.items():
                if similarity >= MIN_SYNONYM_SIMILARITY:
                    if len(synonyms) == 0:
                        synonyms[word] = {word, other_word}
                    elif not word_in_synonyms(word) and not word_in_synonyms(other_word):
                        synonyms[word] = {word, other_word}
                    elif not word_in_synonyms(word):
                        synonyms[other_word].add(word)
                    elif not word_in_synonyms(other_word):
                        synonyms[word].add(other_word)
        if len(synonyms.values()) > 1:
            assert len(set.intersection(*synonyms.values())) == 0
        return list(synonyms.values())

    @staticmethod
    def _get_shortest_word_map(synonyms):
        shortest_word_map = {}
        shortest_word = lambda l: min((word for word in l if word), key=len)
        for synonym_set in synonyms:
            shortest = shortest_word(synonym_set)
            for synonym in set.difference(synonym_set, {shortest}):
                shortest_word_map[synonym] = shortest
        return shortest_word_map

    # From https://stackoverflow.com/questions/17730788/search-and-replace-with-whole-word-only-option
    @staticmethod
    def _homogenize_text(story, word_map, ordered_sentences):
        if len(word_map) == 0:
            return story
        if len(ordered_sentences) >= MIN_NUM_SENT_FOR_PRUNE:
            importances = list(map(itemgetter(1), ordered_sentences))
            importance_1st_quartile = np.percentile(importances, 25)
            pruned_sentences = list(map(itemgetter(0), filter(lambda x: x[1] < importance_1st_quartile, ordered_sentences)))
            for prune in pruned_sentences:
                story = story.replace(prune, '')
        replace = lambda m: word_map[m.group(0)]
        return re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in word_map), replace, story)


def parse_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('-f', '--file', type=str, help='path to text file')
    command_group.add_argument('-a', '--all_files', type=str, help='path to folder with text file')
    command_group.add_argument('-t', '--text', type=str, help='text (use double quotation marks)')
    args = parser.parse_args()
    if args.text:
        return args.text
    if args.file:
        return open(args.file).read()
    if args.all_files:
        path = '{}/{}'.format(args.all_files, args.all_files)
        return open('{}.txt'.format(path)).read()


if __name__ == '__main__':
    text = parse_args()
    preprocessor = Preprocessor(text)
    preprocessor.preprocess()
