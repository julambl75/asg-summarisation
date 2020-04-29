import csv
import os
from operator import itemgetter

from datamuse import datamuse
from pattern.en import conjugate, referenced, lemma

# Format:
# - 2-3 sentence story
#   1. Random verb/tense #1
#   2. Random verb/tense #2 (same for each with probability 0.5)
#   3. Random verb/tense #3 (same for each with probability 0.5)
#   4. Random subject/object #1
#   5. Random linked subject/object #2
#   6. Random linked subject/object #3
# - 1 sentence summary

# action(verb(be, past), subject(matthew, 0, 0), object(logic, a, 0)).
# action(verb(be, past), subject(logic, the, logical), object(0, 0, conjunct(brown, formal))).

PATH = os.path.dirname(os.path.abspath(__file__))


class GenActions:
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

    def generate_actions(self, story_length, num_stories):
        pass


if __name__ == '__main__':
    gen_actions = GenActions()
    gen_actions.generate_actions(2, 100)
