import os
import re
import unicodedata
from operator import itemgetter

import unidecode as unidecode

from gen_data import gen_text

PATH = os.path.dirname(os.path.abspath(__file__))

SOS_TOKEN = 0
EOS_TOKEN = 1

MAX_LENGTH = 100

INPUT = 'stories'
OUTPUT = 'summaries'

TRAIN = 'train'
TEST = 'test'
EVAL = 'eval'


# class Lang:
#     def __init__(self):
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS"}
#         self.n_words = 2  # Count SOS and EOS
#         self.characters = set()
#         self.longest_word = 0
#
#     def add_sentence(self, sentence):
#         for word in sentence.split(' '):
#             self.add_word(word)
#
#     def add_word(self, word):
#         for character in word:
#             self.characters.add(character)
#         if len(word) > self.longest_word:
#             self.longest_word = len(word)
#
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Trim, remove non-letter characters and remove accents
def normalize_string(s):
    s = unicode_to_ascii(s.strip())
    s = unidecode.unidecode(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Read a file and split into lines
def read_data(lang, dataset):
    return open(f'{PATH}/data/{lang}_{dataset}.txt', encoding='utf-8').read().strip().split('\n')


def read_pairs(dataset):
    lines = [read_data(lang, dataset) for lang in (INPUT, OUTPUT)]
    # Split every line into pairs and normalize
    pairs = list(map(lambda pair: tuple(map(normalize_string, pair)), zip(*lines)))
    return pairs


# def read_words():
#     gen_data = gen_text.GenData()
#     words = list(map(itemgetter(0), gen_data.read_words()))
#     names = gen_data.read_names()
#     lang = Lang()
#     for word in words + names:
#         lang.add_word(normalize_string(word))
#     return lang


def prepare_data(dataset):
    pairs = read_pairs(dataset)
    # print("Read %s sentence pairs" % len(pairs))
    # lang = read_words()
    # for pair in pairs:
    #     lang.add_sentence(pair[0])
    #     lang.add_sentence(pair[1])
    # print("Counted words:", lang.n_words)
    return pairs
