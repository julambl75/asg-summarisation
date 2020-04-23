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
        self.bert2emb = {}
        self.emb2bert = []  # dict would allow more efficient lookup, but this is only used for evaluation
        self.n_words = 0

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
        self._add_base_tokens()
        self._read_words()

        # Set initial encodings in BERT universe, get converted to embeddings universe in prepare_embeddings
        self.seq_start_id = self.bert_tokenizer.convert_tokens_to_ids([SEQ_START_TOKEN])[0]
        self.seq_end_id = self.bert_tokenizer.convert_tokens_to_ids([SEQ_END_TOKEN])[0]
        self.seq_pad_id = self.bert_tokenizer.convert_tokens_to_ids([SEQ_PAD_TOKEN])[0]

    def _add_base_tokens(self):
        for base_token in [SEQ_START_TOKEN, SEQ_END_TOKEN, SEQ_PAD_TOKEN, UNKNOWN_TOKEN]:
            self._extend_known_tokens(base_token)

    def _read_words(self):
        gen_data = gen_text.GenData()
        words = list(map(itemgetter(0), gen_data.read_words()))
        names = gen_data.read_names()
        for sequence in words + names:
            self._extend_known_tokens(sequence)

    def _extend_known_tokens(self, sequence):
        bert_ids = self.sequence_to_bert_ids(sequence)
        self.emb2bert.extend(bert_ids)
        return bert_ids

    def sequence_to_bert_ids(self, sequence):
        tokens = self.bert_tokenizer.tokenize(sequence)
        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def bert_ids_to_sequence(self, bert_ids):
        tokens = []
        for bert_id in bert_ids:
            if bert_id in self.bert_tokenizer.ids_to_tokens:
                tokens.append(self.bert_tokenizer.ids_to_tokens[bert_id])
            else:
                tokens.append(UNKNOWN_TOKEN)
        return ' '.join(tokens).replace(' ##', '')

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
        print(f'Read {len(pairs)} sentence pairs from {nn_part} data')
        seq_length = 0
        for pair in pairs:
            for pair_item in pair:
                bert_ids = self._extend_known_tokens(pair_item)
                seq_length = max(seq_length, len(bert_ids))
        return pairs, seq_length

    def prepare_embeddings(self):
        self.emb2bert = sorted(set(self.emb2bert))
        self.bert2emb = {bert_id: emb for emb, bert_id in enumerate(self.emb2bert)}
        self.n_words = len(self.emb2bert)
        print(f'Using embedding with {self.n_words} BERT tokens')
        self.seq_start_id = self.bert2emb[self.seq_start_id]
        self.seq_end_id = self.bert2emb[self.seq_end_id]
        self.seq_pad_id = self.bert2emb[self.seq_pad_id]

        # BERT: 119,000 slots for tokens
        # 1 0 0 0 1 1 ...
        # 1 1 1 ...
