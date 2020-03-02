import argparse
import pprint as pp
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

from ordered_set import OrderedSet
from pycorenlp import StanfordCoreNLP

from parse_concept_net import ParseConceptNet

warnings.filterwarnings("ignore")

IGNORE_POS = []
WEIGHT_SCALE = 10
MIN_SYNONYM_SIMILARITY = 3


class Preprocessor:
    def __init__(self, story):
        self.story = story.lower().strip()

        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.pcn = ParseConceptNet(False)

    def preprocess(self):
        print('Generating POS tags...')
        tokenized = self._tokenize_story()
        pp.pprint(tokenized)
        similar_words, similar_sentences, vocabulary = self._process_similarity(tokenized)

        print('Generating word and sentence adjacency matrices...')
        word_adjacency_mat = self._links_dict_to_adjacency_mat(similar_words, vocabulary)
        sentence_adjacency_mat = self._links_dict_to_adjacency_mat(similar_sentences, range(len(tokenized)))

        print('Plotting similarity between words and sentences')
        labels = {i: vocabulary[i] for i in range(len(vocabulary))}
        self._plot_text_relationship_map(word_adjacency_mat, 'Word Relationship Map', labels)
        self._plot_text_relationship_map(sentence_adjacency_mat, 'Sentence Relationship Map')

        print('Ordering sentences by importance...')
        ordered_sentences = self._order_sentences_by_importance(sentence_adjacency_mat, story)
        pp.pprint(ordered_sentences)

        print('Building synonyms...')
        synonyms = self._get_synonyms(similar_words)
        pp.pprint(synonyms)

        print('Getting shortest words for each set of synonyms...')
        shortest_word_map = self._get_shortest_word_map(synonyms)
        pp.pprint(shortest_word_map)

        print('Homogenizing story using synonyms...')
        homogenized_story = self._homogenize_text(self.story, shortest_word_map)
        pp.pprint(homogenized_story)

        return homogenized_story

    def _tokenize_story(self):
        output = self.nlp.annotate(self.story, properties={
            'annotators': 'pos',
            'outputFormat': 'json'
        })
        tokenized = [[(word['word'], word['pos']) for word in sentence['tokens']] for sentence in output['sentences']]
        return tokenized

    def _process_similarity(self, tokenized):
        similar_words = defaultdict(lambda: defaultdict(lambda: 0))
        similar_sentences = defaultdict(lambda: defaultdict(lambda: 0))
        vocabulary = OrderedSet()

        for i, sentence in enumerate(tokenized):
            for word, pos in sentence:
                for j, other_sentence in enumerate(tokenized):
                    if i != j:
                        for other_word, other_pos in other_sentence:
                            if word != other_word and pos == other_pos and not pos in IGNORE_POS:
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
        graph = nx.from_numpy_matrix(np.matrix(adjacency_mat), create_using=nx.DiGraph)
        layout = nx.circular_layout(graph)
        nx.draw(graph, layout)
        nx.draw_networkx_labels(graph, pos=layout, labels=labels)
        nx.draw_networkx_edge_labels(graph, pos=layout)
        plt.title(title)
        plt.show()

    @staticmethod
    def _order_sentences_by_importance(adjacency_mat, story):
        weights_sum = np.array(list(map(sum, adjacency_mat)))
        importance_ordering = np.argsort(-weights_sum)
        sentences = [sentence.strip() for sentence in story.split('.')]
        ordered_sentences = [sentences[i] + '.' for i in importance_ordering]
        return list(zip(ordered_sentences, sorted(weights_sum, reverse=True)))

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
        if len(synonyms.values()) > 0:
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
    def _homogenize_text(story, word_map):
        if len(word_map) == 0:
            return story
        replace = lambda m: word_map[m.group(0)]
        return re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in word_map), replace, story)


def parse_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('-f', '--file', type=str, help='path to text file')
    command_group.add_argument('-t', '--text', type=str, help='text (use double quotation marks)')
    args = parser.parse_args()
    if args.text:
        return args.text
    if args.file:
        return open(args.file).read()


if __name__ == '__main__':
    story = parse_args()
    preprocessor = Preprocessor(story)
    preprocessor.preprocess()
