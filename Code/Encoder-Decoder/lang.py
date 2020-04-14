import os
import re
import unicodedata
from operator import itemgetter

import contractions
from pytorch_pretrained_bert import BertTokenizer
from unidecode import unidecode

from gen_data import gen_text

PATH = os.path.dirname(os.path.abspath(__file__))

SEQ_START_TOKEN = '[CLS]'
SEQ_END_TOKEN = '[SEP]'
SEQ_PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'

INPUT = 'stories'
OUTPUT = 'summaries'

TRAIN = 'train'
TEST = 'test'
EVAL = 'eval'


class Lang:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
        self._read_words()

        self.seq_start_id = self.tokenizer.convert_tokens_to_ids([SEQ_START_TOKEN])[0]
        self.seq_end_id = self.tokenizer.convert_tokens_to_ids([SEQ_END_TOKEN])[0]
        self.seq_pad_id = self.tokenizer.convert_tokens_to_ids([SEQ_PAD_TOKEN])[0]

    def _read_words(self):
        gen_data = gen_text.GenData()
        words = list(map(itemgetter(0), gen_data.read_words()))
        names = gen_data.read_names()
        for sequence in words + names:
            self.sequence_to_ids(sequence)

    def sequence_to_ids(self, sequence):
        tokens = self.tokenizer.tokenize(sequence)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def id_to_sequence(self, id):
        if id in self.tokenizer.ids_to_tokens:
            return self.tokenizer.ids_to_tokens[id]
        return UNKNOWN_TOKEN


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Trim, expand contractions and add start/end markers
def format_sequence(s):
    s = unicode_to_ascii(s.strip())
    s = contractions.fix(s)
    return f'{SEQ_START_TOKEN} {s} {SEQ_END_TOKEN}'


# Read a file and split into lines
def read_data(lang, dataset):
    return open(f'{PATH}/data/{lang}_{dataset}.txt', encoding='utf-8').read().strip().split('\n')


def read_pairs(dataset):
    lines = [read_data(pair_item, dataset) for pair_item in (INPUT, OUTPUT)]
    # Split every line into pairs and normalize
    pairs = list(map(lambda pair: tuple(map(format_sequence, pair)), zip(*lines)))
    return pairs


def prepare_data(dataset, lang):
    pairs = read_pairs(dataset)
    print("Read %s sentence pairs" % len(pairs))
    max_seq_length = 0
    for pair in pairs:
        for pair_item in pair:
            ids = lang.sequence_to_ids(pair_item)
            max_seq_length = max(max_seq_length, len(ids))
    return pairs, max_seq_length
