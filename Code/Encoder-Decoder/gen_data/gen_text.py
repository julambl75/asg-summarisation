# 'The flowers are blooming now' # DT NNS V VING ADV
# 'The leaves are growing'
# 'Spring is coming very soon'
# -> 'Spring is coming'
#
# 'The wall is covered in poison ivy' # DT NN V VING IN NN
# 'At night the house is full of ghosts'
# 'You can hear voices coming from inside'
# 'The house must be haunted'
# -> 'At night the house is full of ghosts so it is haunted'
#
# 'Suzanne went to cooking school'
# 'Sophie learned to cook in Australia'
# 'They are owners of a famous restaurant in Melbourne'
# -> 'Sophie and Suzanne are renowned chefs' # NNP CC NNP V ADJ NNS

### Possesion:
# Joe/dog/rambunctious
# -> Joe had a rambunctious dog.
# Joe had a dog.
# The [hound] was rambunctious and [little].

### Doing something (work/sit/stand):
# TODO

### Getting dressed: Sarah put on her cape and make a phone call. / Peter put on his sweatshirt and went to the gym.
# TODO

###
# We need a large number of pairs for train (~9,000), a small number for test (~1,000), and a few for eval (~10).
###
import csv
import os
import random

from operator import itemgetter
from pattern.en import conjugate, singularize, pluralize, referenced, lemma
from datamuse import datamuse

PATH = os.path.dirname(os.path.abspath(__file__))

NOUN = 'n'
VERB = 'v'
ADJ = 'j'


class GenData:
    def __init__(self):
        self.words = self.read_words()
        self.names = self.read_names()

        self.nouns = self.get_words_of_type('n')
        self.adjectives = self.get_words_of_type('j')

        self.datamuse_api = datamuse.Datamuse()
        # There is a bug with Python 3.7 causing the first call to Pattern to crash due to a StopIteration
        try:
            lemma('eight')
        except:
            pass

    @staticmethod
    def read_names():
        with open(f'{PATH}/words/names.txt', encoding='utf-8') as names_file:
            return names_file.read().strip().split('\n')

    @staticmethod
    def read_words():
        with open(f'{PATH}/words/words.csv') as words_csv:
            reader = csv.reader(words_csv, delimiter=',')
            return [tuple(row) for row in reader]

    def get_words_of_type(self, word_type):
        return list(map(itemgetter(0), filter(lambda e: e[1] == word_type, self.words)))

    @staticmethod
    def get_random(words):
        return random.choice(words)

    def find_common_adj_for_noun(self, noun):
        words = self.datamuse_api.words(rel_jjb=noun, max=5)
        return self.get_random(words)['word'] if words else None

    def find_synonym_with_context(self, noun, context):
        words = self.datamuse_api.words(rel_syn=noun, topics=context, max=5)
        return self.get_random(words)['word'] if words else None

    def make_summary(self, subject, descriptor, adjective):
        other_descriptor = self.find_synonym_with_context(descriptor, adjective)
        other_adjective = self.find_common_adj_for_noun(descriptor)
        if other_descriptor is None or other_adjective is None:
            return None
        verb = conjugate('be', tense='past', person=3, number='singular')
        s_object_ind = referenced(descriptor, article='indefinite')
        s_object_def = referenced(other_descriptor, article='definite')
        s_object_adj = referenced(' '.join([adjective, descriptor]), article='indefinite')
        summary = ' '.join([subject, verb, s_object_adj])
        sentence1 = ' '.join([subject, verb, s_object_ind])
        sentence2 = ' '.join([s_object_def, verb, adjective, 'and', other_adjective])
        return summary, [sentence1, sentence2]


# https://www.clips.uantwerpen.be/pages/pattern-en
# https://www.datamuse.com/api/
if __name__ == '__main__':
    gen_data = GenData()

    for i in range(5):
        print(gen_data.make_summary('Joe', 'dog', 'rambunctious'))
    for i in range(10):
        subject = gen_data.get_random(gen_data.names)
        descriptor = gen_data.get_random(gen_data.nouns)
        adjective = gen_data.get_random(gen_data.adjectives)
        print(gen_data.make_summary(subject, descriptor, adjective))
