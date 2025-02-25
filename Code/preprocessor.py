import argparse
import math
import pprint as pp
import re
import warnings
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ordered_set import OrderedSet

from parse_concept_net import ParseConceptNet
from simplify_tokenize_text import TextSimplifier

warnings.filterwarnings("ignore")

IGNORE_POS = ['DT', '.']
SAME_WORD_POS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_POS = 'VB'
ADJECTIVE_POS = 'JJ'
ADVERB_POS = 'RB'

SAME_WORD_SIMILARITY = 5
WEIGHT_SCALE = 10
MIN_SYNONYM_SIMILARITY = 2
SENT_IMPORTANCE_SQRT = 10
MIN_NUM_SENT_FOR_PRUNE = 3


# Note: after pre-processing a text, the word order may change, but capitalisation does not
class Preprocessor:
    def __init__(self, story, print_results=True):
        self.story = story.strip().replace('\n', ' ')
        self.print_results = print_results

        self.proper_nouns = set()

        self.tokenizer = TextSimplifier(self.story)
        self.pcn = ParseConceptNet(False)

    def preprocess(self):
        if self.print_results:
            pp.pprint(self.story)
        tokenized, self.story, self.proper_nouns = self.tokenizer.tokenize()

        if self.print_results:
            print('\nGenerating POS tags and simplifying story...')
            pp.pprint(tokenized)
            pp.pprint(self.story)
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

        homogenised_story = self._homogenise_text(self.story, shortest_word_map, ordered_sentences)
        if self.print_results:
            print('\nHomogenizing story using synonyms...')
            pp.pprint(homogenised_story)

        return homogenised_story, self.proper_nouns

    @staticmethod
    def _similar_pos(pos, other_pos):
        if pos in SAME_WORD_POS and other_pos in SAME_WORD_POS:
            return True
        if VERB_POS in pos and VERB_POS in other_pos:
            return True
        if ADJECTIVE_POS in pos and ADJECTIVE_POS in other_pos:
            return True
        if ADVERB_POS in pos and ADVERB_POS in other_pos:
            return True
        if pos in IGNORE_POS or other_pos in IGNORE_POS:
            return False
        return pos == other_pos

    def _process_similarity(self, tokenized):
        similar_words = defaultdict(lambda: defaultdict(lambda: 0))
        similar_sentences = defaultdict(lambda: defaultdict(lambda: 0))
        similarity_cache = defaultdict(lambda: defaultdict(lambda: int))
        vocabulary = OrderedSet()

        for i, sentence in enumerate(tokenized):
            for word, pos in sentence:
                for j, other_sentence in enumerate(tokenized):
                    for other_word, other_pos in other_sentence:
                        if i != j and self._similar_pos(pos, other_pos):
                            word = word.lower()
                            other_word = other_word.lower()

                            # Similarity already computed
                            if other_word in similarity_cache[word].keys():
                                similar_sentences[i][j] += similarity_cache[word][other_word]
                            elif word in similarity_cache[other_word].keys():
                                similar_sentences[i][j] += similarity_cache[other_word][word]
                            else:
                                similarity = lemma_similarity = 0
                                if word == other_word:
                                    similarity = SAME_WORD_SIMILARITY if pos in SAME_WORD_POS else 0
                                elif pos == other_pos and not pos.startswith(VERB_POS):
                                    similarity = self.pcn.compare_words(word, other_word)
                                if not similarity:
                                    lemma_similarity = self.pcn.compare_words(word, other_word, use_lemma=True)
                                max_similarity = max(similarity, lemma_similarity)
                                if pos == other_pos and word != other_word and max_similarity:
                                    vocabulary.add(word)
                                    vocabulary.add(other_word)
                                    similar_words[word][other_word] = max_similarity
                                similarity_cache[word][other_word] = max_similarity
                                similar_sentences[i][j] += max_similarity
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
    def _homogenise_text(story, word_map, ordered_sentences):
        if len(ordered_sentences) >= MIN_NUM_SENT_FOR_PRUNE:
            importances = list(map(itemgetter(1), ordered_sentences))
            importance_1st_quartile = np.percentile(importances, 25)
            pruned_sentences = list(map(itemgetter(0), filter(lambda x: x[1] < importance_1st_quartile, ordered_sentences)))
            for prune in pruned_sentences:
                story = story.replace(prune, '')

        if len(word_map) == 0:
            return story
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
