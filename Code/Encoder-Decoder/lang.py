import os
import unicodedata
from operator import itemgetter

import contractions
from pytorch_pretrained_bert import BertTokenizer

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
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

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
            self.sequence_to_bert_ids(sequence)

    def sequence_to_bert_ids(self, sequence):
        tokens = self.tokenizer.tokenize(sequence)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def bert_id_to_sequence(self, bert_id):
        if bert_id in self.tokenizer.ids_to_tokens:
            return self.tokenizer.ids_to_tokens[bert_id]
        return UNKNOWN_TOKEN

    # class Lang:
    #
    #     def add_sentence(self, sentence):
    #         for word in sentence.split(' '):
    #             self.add_word(word)
    #
    #         if word not in self.word2index:
    #             self.word2index[word] = self.n_words
    #             self.index2word[self.n_words] = word
    #             self.n_words += 1

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Trim, expand contractions and add start/end markers
    def format_sequence(self, s):
        s = self.unicode_to_ascii(s.strip())
        s = contractions.fix(s)
        return f'{SEQ_START_TOKEN} {s} {SEQ_END_TOKEN}'

    # Read a file and split into lines
    @staticmethod
    def read_data(dataset, nn_part):
        return open(f'{PATH}/data/{dataset}_{nn_part}.txt', encoding='utf-8').read().strip().split('\n')

    def read_pairs(self, nn_part):
        lines = [self.read_data(pair_item, nn_part) for pair_item in (INPUT, OUTPUT)]
        # Split every line into pairs and normalize
        pairs = list(map(lambda pair: tuple(map(self.format_sequence, pair)), zip(*lines)))
        return pairs

    def prepare_data(self, nn_part):
        pairs = self.read_pairs(nn_part)
        print("Read %s sentence pairs" % len(pairs))

        seq_length = 0
        for pair in pairs:
            for pair_item in pair:
                ids = self.sequence_to_bert_ids(pair_item)
                seq_length = max(seq_length, len(ids))

        bert_ids = set(itertools.chain.from_iterable(
            [list(itertools.chain.from_iterable(tp1.tolist() + tp2.tolist())) for tp1, tp2 in training_pairs]))
        bert_ids.add(100)  # [UNK]
        bert_ids_sorted = sorted(list(bert_ids))
        bert_to_emb = {bert_id: emb for emb, bert_id in enumerate(bert_ids_sorted)}

        return pairs, seq_length
