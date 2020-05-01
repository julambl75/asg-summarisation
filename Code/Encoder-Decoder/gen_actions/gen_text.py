import csv
import os
import random
from operator import itemgetter

from datamuse import datamuse
from pattern.en import conjugate, referenced, lemma
from pattern.en import PRESENT, PAST, GERUND

# Format:
# - 2-3 sentence story
#   1. Random verb/tense #1
#   2. Random verb/tense #2 (same for each with probability 0.5)
#   3. Random verb/tense #3 (same for each with probability 0.5)
#   4. Random subject/object #1
#   5. Random linked subject/object #2
#   6. Random linked subject/object #3
# - 1 sentence summary

# action(0, verb(be, past), subject(matthew, 0, 0), object(logic, a, 0)).
# action(1, verb(be, past), subject(logic, the, logical), object(0, 0, conjunct(brown, formal))).

PATH = os.path.dirname(os.path.abspath(__file__))
WORDS_PATH = os.path.dirname(PATH)

DETERMINERS = ['a', 'the']
PRONOUNS = ['I', 'you', 'he', 'she', 'it', 'we', 'they']
TENSES = ['present', 'present_third', 'past', 'gerund']

MIN_ADJ, MAX_ADJ = (0, 2)
PROB_PRONOUN = 0.5


class GenActions:
    def __init__(self):
        self.words = self.read_words()
        self.names = self.read_names()

        self.verbs = self.get_words_of_type('v')
        self.nouns = self.get_words_of_type('n')
        self.adjectives = self.get_words_of_type('j')

        self.datamuse_api = datamuse.Datamuse()
        # # There is a bug with Python 3.7 causing the first call to Pattern to crash due to a StopIteration
        try:
            lemma('eight')
        except:
            pass

    @staticmethod
    def read_names():
        with open(f'{WORDS_PATH}/words/names.txt', encoding='utf-8') as names_file:
            return names_file.read().strip().split('\n')

    @staticmethod
    def read_words():
        with open(f'{WORDS_PATH}/words/words.csv') as words_csv:
            reader = csv.reader(words_csv, delimiter=',')
            return [tuple(row) for row in reader]

    def get_words_of_type(self, word_type):
        return list(map(itemgetter(0), filter(lambda e: e[1] == word_type, self.words)))

    def get_random_subject(self, person=False):
        if person:
            noun = random.choice(PRONOUNS if random.random() < PROB_PRONOUN else self.names)
        else:
            determiner = random.choice(DETERMINERS)
            noun = random.choice(self.nouns)
            num_adj = random.randint(MIN_ADJ, MAX_ADJ)
            adjectives = [random.choice(self.adjectives) for _ in range(num_adj)]
        pass

    def get_random_verb(self, third_person=True):
        verb = random.choice(self.verb)

    def get_random_object(self, person=False):
        pass

    def generate_action(self, index):
        subject = self.get_random_subject()
        verb = self.get_random_verb()
        object = self.get_random_object()
        return None

    def generate_stories(self, story_length, num_stories):
        stories = []
        for _ in range(num_stories):
            stories.append([self.generate_action(i) for i in range(story_length)])
        return


if __name__ == '__main__':
    gen_actions = GenActions()
    gen_actions.generate_stories(2, 100)
    raise Exception('TOOD generate leaf nodes for ASG')
