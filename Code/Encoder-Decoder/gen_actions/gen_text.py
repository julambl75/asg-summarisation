import csv
import os
import random
from operator import itemgetter

from datamuse import datamuse
from pattern.en import conjugate, lemma
from pattern.en import SG, PL, PRESENT

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
PRONOUNS = {('I', 1, SG), ('you', 2, SG), ('he', 3, SG), ('she', 3, SG), ('it', 3, SG),
            ('we', 1, PL), ('you', 2, PL), ('they', 3, PL)}
TENSES = ['present', 'past']
TENSE_THIRD_POSTFIX = '_third'

PROB_PERSON = 0.5
PROB_PRONOUN = 0.5
MIN_ADJ, MAX_ADJ = (0, 2)
MIN_PEOPLE, MAX_PEOPLE = (1, 2)

EMPTY_TOKEN = '0'
CONJUNCT_TOKEN = 'conjunct'
VERB_TOKEN = 'verb'
SUBJECT_TOKEN = 'subject'
OBJECT_TOKEN = 'object'


class GenActions:
    def __init__(self):
        self.words = self.read_words()
        self.names = self.read_names()

        self.verbs = self.get_words_of_type('v')
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
        with open(f'{WORDS_PATH}/words/names.txt', encoding='utf-8') as names_file:
            return names_file.read().strip().split('\n')

    @staticmethod
    def read_words():
        with open(f'{WORDS_PATH}/words/words.csv') as words_csv:
            reader = csv.reader(words_csv, delimiter=',')
            return [tuple(row) for row in reader]

    def get_words_of_type(self, word_type):
        return list(map(itemgetter(0), filter(lambda e: e[1] == word_type, self.words)))

    @staticmethod
    def _list_to_conjunct(tokens):
        if len(tokens) == 0:
            return EMPTY_TOKEN
        elif len(tokens) == 1:
            return tokens[0]
        return f'{CONJUNCT_TOKEN}({", ".join(tokens)})'

    def get_random_subject_object(self, subject_or_object):
        if random.random() < PROB_PERSON:
            nouns = []
            for _ in random.randint(MIN_PEOPLE, MAX_PEOPLE):
                if random.random() < PROB_PRONOUN:
                    pronoun, person, number = random.choice(PRONOUNS)
                    nouns.append(pronoun)
                else:
                    name = random.choice(self.names)
            noun = self._list_to_conjunct(nouns)
            determiner = EMPTY_TOKEN
            adjective_part = EMPTY_TOKEN
        else:
            determiner = random.choice(DETERMINERS)
            noun = random.choice(self.nouns)
            num_adjectives = random.randint(MIN_ADJ, MAX_ADJ)
            adjectives = [random.choice(self.adjectives) for _ in range(num_adjectives)]
            adjective_part = self._list_to_conjunct(adjectives)
        return f'{subject_or_object}({noun}, {determiner}, {adjective_part})'

    def get_random_verb(self, person, number):
        verb = random.choice(self.verbs)
        tense = random.choice(TENSES)
        conjugated = conjugate(verb, person, tense=tense, number=number)
        if tense == PRESENT and person == 3:
            tense += TENSE_THIRD_POSTFIX
        return f'verb({verb}, {tense})'

    def generate_action(self, index):
        subject = self.get_random_subject_object(SUBJECT_TOKEN)
        verb = self.get_random_verb()
        object = self.get_random_subject_object(OBJECT_TOKEN)
        return f'action({index}, {verb}, {subject}, {object})'

    def generate_stories(self, story_length, num_stories):
        story_actions = []
        for _ in range(num_stories):
            story_actions.append([self.generate_action(i) for i in range(story_length)])
        return story_actions


if __name__ == '__main__':
    gen_actions = GenActions()
    stories = gen_actions.generate_stories(2, 5)
    raise Exception('TOOD generate leaf nodes for ASG')
