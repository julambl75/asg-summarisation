import csv
import os
import random
from operator import itemgetter

from datamuse import datamuse
from pattern.en import conjugate, lemma
from pattern.en import SG, PL, PRESENT

# Format:
# - 3 sentence story
#   1. Random verb/tense #1
#   2. Random verb/tense #2 (same for each with probability 0.5)
#   3. Random verb/tense #3 (same for each with probability 0.5)
#   4. Random subject/object #1
#   5. Random linked subject/object #2
#   6. Random linked subject/object #3
# - 1-2 sentence summary

# action(0, verb(be, past), subject(matthew, 0, 0), object(logic, a, 0)).
# action(1, verb(be, past), subject(logic, the, logical), object(0, 0, conjunct(brown, formal))).

PATH = os.path.dirname(os.path.abspath(__file__))
WORDS_PATH = os.path.dirname(PATH)

DETERMINERS = ['a', 'the']
DETERMINER_REOCCUR = 'the'
PRONOUNS_SUBJECT = [('I', 1, SG), ('you', 2, SG), ('he', 3, SG), ('she', 3, SG), ('it', 3, SG),
                    ('we', 1, PL), ('you', 2, PL), ('they', 3, PL)]
SUBJECT_TO_OBJECT = {'I': 'me', 'he': 'him', 'she': 'her', 'we': 'us', 'they': 'them'}
OBJECT_TO_SUBJECT = {v: k for k, v in SUBJECT_TO_OBJECT.items()}
DEFAULT_VERB = 'be'

PROB_PERSON = 0.5
PROB_PRONOUN = 0.5
PROB_LAST_NOUN = 0.25
PROB_DEFAULT_VERB = 0.5
MIN_ADJ, MAX_ADJ = (0, 2)
MIN_PEOPLE, MAX_PEOPLE = (1, 2)

CONJUGATION_INDIVIDUAL = (3, SG)
CONJUGATION_GROUP = (3, PL)

EMPTY_TOKEN = '0'
VERB_TOKEN = 'verb'
SUBJECT_TOKEN = 'subject'
OBJECT_TOKEN = 'object'
CONJUNCT_TOKEN = 'conjunct'

TENSES = ['present', 'past']
TENSE_THIRD_POSTFIX = '_third'
VERB_PREDICATE = 'verb'
NOUN_PREDICATE = 'noun'
DETERMINER_PREDICATE = 'det'
ADJECTIVE_PREDICATE = 'adj_or_adv'
PRONOUN_PREDICATE = 'prp'


class GenActions:
    def __init__(self):
        self.words = self.read_words()
        self.names = self.read_names()

        self.verbs = self.get_words_of_type('v')
        self.nouns = self.get_words_of_type('n')
        self.adjectives = self.get_words_of_type('j')

        self._reset_last_nouns()
        self.leaf_nodes = set()

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

    def _reset_last_nouns(self):
        self.last_nouns = {OBJECT_TOKEN: [], SUBJECT_TOKEN: []}

    def _get_random_last_noun(self, token_type):
        noun, determiner, adjective_part, person, number = random.choice(self.last_nouns[token_type])
        if token_type == OBJECT_TOKEN:
            if determiner != EMPTY_TOKEN:
                determiner = DETERMINER_REOCCUR
            if noun in SUBJECT_TO_OBJECT.keys():
                noun = SUBJECT_TO_OBJECT[noun]
        elif token_type == SUBJECT_TOKEN and noun in OBJECT_TO_SUBJECT.keys():
            noun = OBJECT_TO_SUBJECT[noun]
        return noun, determiner, adjective_part, person, number

    def _get_random_person(self, token_type):
        pronouns = PRONOUNS_SUBJECT
        nouns = []
        num_people = random.randint(MIN_PEOPLE, MAX_PEOPLE)
        for _ in range(num_people):
            if random.random() < PROB_PRONOUN:
                pronoun, person, number = random.choice(pronouns)
                if token_type == OBJECT_TOKEN and pronoun in SUBJECT_TO_OBJECT.keys():
                    pronoun = SUBJECT_TO_OBJECT[pronoun]
                nouns.append(pronoun)
            else:
                name = random.choice(self.names)
                person, number = CONJUGATION_INDIVIDUAL
                nouns.append(name)
        if num_people > 1:
            person, number = CONJUGATION_GROUP
        noun = self._list_to_conjunct(nouns)
        determiner = EMPTY_TOKEN
        adjective_part = EMPTY_TOKEN
        return noun, determiner, adjective_part, person, number

    def _get_random_common_noun(self):
        person, number = CONJUGATION_INDIVIDUAL
        num_adjectives = random.randint(MIN_ADJ, MAX_ADJ)
        determiner = random.choice(DETERMINERS)
        noun = random.choice(self.nouns)
        adjectives = [random.choice(self.adjectives) for _ in range(num_adjectives)]
        adjective_part = self._list_to_conjunct(adjectives)
        return noun, determiner, adjective_part, person, number

    def get_random_subject_object(self, token_type):
        if self.last_nouns[token_type] and random.random() < PROB_LAST_NOUN:
            noun, determiner, adjective_part, person, number = self._get_random_last_noun(token_type)
        elif random.random() < PROB_PERSON:
            noun, determiner, adjective_part, person, number = self._get_random_person(token_type)
        else:
            noun, determiner, adjective_part, person, number = self._get_random_common_noun()
        self.last_nouns[token_type].append((noun, determiner, adjective_part, person, number))
        self.create_asg_leaf()
        token = f'{token_type}({noun}, {determiner}, {adjective_part})'
        if token_type == SUBJECT_TOKEN:
            return token, person, number
        return token

    def get_random_verb(self, person, number):
        if random.random() < PROB_DEFAULT_VERB:
            verb = random.choice(self.verbs)
        else:
            verb = DEFAULT_VERB
        tense = random.choice(TENSES)
        conjugated = conjugate(verb, person, tense=tense, number=number)
        if tense == PRESENT and person == 3:
            tense += TENSE_THIRD_POSTFIX
        return f'verb({verb}, {tense})'
    
    @staticmethod
    def create_asg_leaf(pos_tag, value, predicate):
        if predicate == VERB_PREDICATE:
            verb_name = None
            verb_form = None
            lemma = f'{verb_name}, {verb_form}'
        else:
            name = None
            lemma = name.lower()
        return f'{pos_tag} -> "{value} " {{ {predicate}({value}). }}'

    def generate_action(self, index):
        subject, person, number = self.get_random_subject_object(SUBJECT_TOKEN)
        verb = self.get_random_verb(person, number)
        object = self.get_random_subject_object(OBJECT_TOKEN)
        return f'action({index}, {verb}, {subject}, {object})'

    def generate_stories(self, story_length, num_stories):
        self.leaf_nodes = set()
        story_actions = []
        for _ in range(num_stories):
            story_actions.append([self.generate_action(i) for i in range(story_length)])
            self._reset_last_nouns()
        return story_actions, self.leaf_nodes


if __name__ == '__main__':
    gen_actions = GenActions()
    stories, asg_leaves = gen_actions.generate_stories(3, 5)
    raise Exception('TOOD generate leaf nodes for ASG')
