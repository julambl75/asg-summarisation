import re
import unicodedata

SOS_TOKEN = 0
EOS_TOKEN = 1

MAX_LENGTH = 100

INPUT = 'stories'
OUTPUT = 'summaries'

TRAIN = 'train'
TEST = 'test'
EVAL = 'eval'


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Read a file and split into lines
def read_data(lang, dataset):
    return open(f'data/{lang}_{dataset}.txt', encoding='utf-8').read().strip().split('\n')


def read_langs(dataset):
    print("Reading lines...")
    lines = [read_data(lang, dataset) for lang in (INPUT, OUTPUT)]

    # Split every line into pairs and normalize
    pairs = list(map(lambda pair: tuple(map(normalize_string, pair)), zip(*lines)))

    input_lang = Lang(INPUT)
    output_lang = Lang(OUTPUT)

    return input_lang, output_lang, pairs


# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s ",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )
#
#
# def filter_pair(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#            len(p[1].split(' ')) < MAX_LENGTH and \
#            p[1].startswith(eng_prefixes)
#
#
# def filter_pairs(pairs):
#     return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(dataset):
    input_lang, output_lang, pairs = read_langs(dataset)
    print("Read %s sentence pairs" % len(pairs))
    # pairs = filter_pairs(pairs)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
