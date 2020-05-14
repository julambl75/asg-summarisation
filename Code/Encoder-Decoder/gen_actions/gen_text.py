import csv
import os
import random
from distutils.dir_util import mkpath
from operator import itemgetter

import language_check
from datamuse import datamuse
from pattern.en import SG, PL, PRESENT

from query_pattern import QueryPattern
from score_summary import SummaryScorer
from text_to_summary import TextToSummary

PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PATH)

EXPORT_PATH = f'{PARENT_DIR}/data'
NAMES_FILE = f'{PARENT_DIR}/words/names.txt'
WORDS_FILE = f'{PARENT_DIR}/words/words.csv'

TRAIN = 'train'
TEST = 'test'
EVAL = 'eval'

DETERMINERS = ['a', 'the']
DETERMINER_REOCCUR = 'the'
CONJUNCT_KEYWORD = 'and'
PRONOUNS_SUBJECT = [('I', 1, SG), ('you', 2, SG), ('he', 3, SG), ('she', 3, SG), ('it', 3, SG),
                    ('we', 1, PL), ('you', 2, PL), ('they', 3, PL)]
SUBJECT_TO_OBJECT = {'I': 'me', 'he': 'him', 'she': 'her', 'we': 'us', 'they': 'them'}
OBJECT_TO_SUBJECT = {v: k for k, v in SUBJECT_TO_OBJECT.items()}
DEFAULT_VERB = 'be'
PUNCTUATION = '.'

PROB_PERSON = 0#.5 TODO
PROB_PRONOUN = 0#.5 TODO
PROB_LAST_NOUN = 0.25
MIN_ADJ, MAX_ADJ = (0, 1)
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

COMMON_NOUN_POS = 'nn'
PROPER_NOUN_POS = 'nnp'
PRONOUN_POS = 'prp'
DETERMINER_POS = 'dt'
ADJECTIVE_POS = 'jj'
TENSE_TO_POS_TAG = {'present': 'vbp', 'present_third': 'vbz', 'past': 'vbd'}

PRINT_EVERY_ITERS = 50
TEST_PROPORTION = 0.1


# Idea:
# 1. Pick random common/proper noun(s)
# 2. Find common verb/adjective for it
#  -> make sentence
# 3. Pick noun in same lexical field if common otherwise same
# 4. Find common verb/adjective for it
#  -> make sentence
# 5. Add irrelevant sentence

# words = self.datamuse_api.words(rel_syn=noun, topics=context, max=1)


class GenActions:
    def __init__(self):
        self.words = self.read_words()
        self.names = self.read_names()

        self.verbs = self.get_words_of_type('v')
        self.nouns = self.get_words_of_type('n')
        self.adjectives = self.get_words_of_type('j')

        self.proper_nouns = set()
        self._reset_for_new_story()

        self.story_actions = []
        self.story_leaf_nodes = []
        self.stories = []
        self.training_pairs = []

        self.query_pattern = QueryPattern()
        self.datamuse_api = datamuse.Datamuse()
        self.language_checker = language_check.LanguageTool('en-GB')
        self.summary_scorer = SummaryScorer()

    @staticmethod
    def read_names():
        with open(NAMES_FILE, encoding='utf-8-sig') as names_file:
            return names_file.read().strip().split('\n')

    @staticmethod
    def read_words():
        with open(WORDS_FILE) as words_csv:
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

    def _reset_for_new_story(self):
        self.last_nouns = {OBJECT_TOKEN: [], SUBJECT_TOKEN: []}
        self.leaf_nodes = set()
        self.curr_story_tokens = []

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
                self.create_asg_leaf(PRONOUN_POS, pronoun, NOUN_PREDICATE)
            else:
                name = random.choice(self.names)
                person, number = CONJUGATION_INDIVIDUAL
                nouns.append(name)
                self.proper_nouns.add(name)
                self.create_asg_leaf(PROPER_NOUN_POS, name, NOUN_PREDICATE)
        if num_people > 1:
            person, number = CONJUGATION_GROUP
        noun = self._list_to_conjunct(nouns)
        determiner = EMPTY_TOKEN
        adjective_part = EMPTY_TOKEN
        return noun, determiner, adjective_part, person, number

    def _get_random_common_noun(self):
        person, number = CONJUGATION_INDIVIDUAL
        determiner = random.choice(DETERMINERS)
        noun = random.choice(self.nouns)

        # TODO cleanup
        adjectives = [x['word'] for x in self.datamuse_api.words(rel_jjb=noun, max=1) if 'word' in x.keys() and x['word'] in self.adjectives]
        adjective = self._list_to_conjunct(adjectives)

        # TODO rel_trg, rel_gen, rel_syn
        # words = [x['word'] for x in self.datamuse_api.words(rel='participation', max=10) if 'word' in x.keys()]
        # [w for w in words if w in self.nouns]
        # [w for w in words if w in self.adjectives]

        self.create_asg_leaf(COMMON_NOUN_POS, noun, NOUN_PREDICATE)
        self.create_asg_leaf(DETERMINER_POS, determiner, DETERMINER_PREDICATE)
        self.create_asg_leaf(ADJECTIVE_POS, adjective, ADJECTIVE_PREDICATE)
        return noun, determiner, adjective, person, number

    def _subject_object_to_story_tokens(self, noun, determiner, adjective_part):
        for clause_part in [determiner, adjective_part, noun]:
            inner_token_idx = clause_part.find('(') + 1
            if inner_token_idx > 0:
                inner_tokens = clause_part[inner_token_idx:-1].split(',')
                for i, inner_token in enumerate(inner_tokens):
                    if i > 0:
                        self.curr_story_tokens.append(CONJUNCT_KEYWORD)
                    self.curr_story_tokens.append(inner_token.strip())
            elif clause_part != EMPTY_TOKEN:
                self.curr_story_tokens.append(clause_part)

    # TODO cleanup
    def get_random_subject_object(self, token_type):
        if self.last_nouns[token_type] and random.random() < PROB_LAST_NOUN:
            noun, determiner, adjective_part, person, number = self._get_random_last_noun(token_type)
        elif random.random() < PROB_PERSON:
            noun, determiner, adjective_part, person, number = self._get_random_person(token_type)
        else:
            noun, determiner, adjective_part, person, number = self._get_random_common_noun()
        self.last_nouns[token_type].append((noun, determiner, adjective_part, person, number))
        self._subject_object_to_story_tokens(noun, determiner, adjective_part)
        token = f'{token_type}({noun}, {determiner}, {adjective_part})'
        if token_type == SUBJECT_TOKEN:
            return token, noun, adjective_part, person, number
        return token

    # TODO cleanup
    def get_random_verb(self, noun, adjective, person, number):
        # if random.random() < PROB_DEFAULT_VERB:
        #     verb = random.choice(self.verbs)
        # else:
        #     verb = DEFAULT_VERB
        verbs = [x['word'] for x in self.datamuse_api.words(rel_syn=noun, topics=adjective, max=10) if 'word' in x.keys() and x['word'] in self.verbs]
        verb = verbs[0] if verbs else DEFAULT_VERB

        tense = random.choice(TENSES)
        conjugated = self.query_pattern.conjugate(verb, person=person, tense=tense, number=number)
        if tense == PRESENT and person == 3:
            tense += TENSE_THIRD_POSTFIX
        self.create_asg_leaf(TENSE_TO_POS_TAG[tense], conjugated, VERB_PREDICATE, verb, tense)
        self.curr_story_tokens.append(conjugated)
        return f'verb({verb}, {tense})'

    def create_asg_leaf(self, pos_tag, value, predicate, verb_name=None, verb_form=None):
        if predicate == VERB_PREDICATE:
            lemma = f'{verb_name}, {verb_form}'
        else:
            lemma = value.lower().replace('-', '_')
        leaf_node = f'{pos_tag} -> "{value} " {{ {predicate}({lemma}). }}'
        self.leaf_nodes.add(leaf_node)

    def generate_action(self, index):
        subject, noun, adjective, person, number = self.get_random_subject_object(SUBJECT_TOKEN)
        verb = self.get_random_verb(noun, adjective, person, number)
        object = self.get_random_subject_object(OBJECT_TOKEN)
        self.curr_story_tokens.append(PUNCTUATION)
        return f'action({index}, {verb}, {subject}, {object}).'.replace('-', '_').lower()

    def format_story(self):
        joined_tokens = ' '.join(self.curr_story_tokens)
        return self.language_checker.correct(joined_tokens)

    def generate_stories(self, story_length, num_stories):
        for i in range(num_stories):
            if i % PRINT_EVERY_ITERS == 0:
                print(f'[{i}/{num_stories}]: Generating stories of length {story_length}...')
            self.story_actions.append([self.generate_action(i) for i in range(story_length)])
            self.story_leaf_nodes.append(sorted(self.leaf_nodes))
            self.stories.append(self.format_story())
            self._reset_for_new_story()
        print(f'[{num_stories}/{num_stories}]: Generated stories of length {story_length}...')

    def summarise_generated_stories(self):
        num_stories = len(self.story_actions)

        for action_set, leaf_node_set, story in zip(self.story_actions, self.story_leaf_nodes, self.stories):
            if len(self.training_pairs) % PRINT_EVERY_ITERS == 0:
                print(f'[{len(self.training_pairs)}/{num_stories}]: Summarising generated stories...')

            text_to_summary = TextToSummary(story, gen_actions.proper_nouns, print_results=False)
            summaries = text_to_summary.generate_summaries(action_set, (leaf_node_set,))

            self.training_pairs.append((story, summaries[0][0]))
        print(f'[{num_stories}/{num_stories}]: Summarised generated stories...')

    @staticmethod
    def get_export_file(data_type, nn_step):
        return f'{EXPORT_PATH}/{data_type}_{nn_step}.txt'

    def _write_training_data(self, story_summary_pairs, nn_step):
        print(f'Writing {len(story_summary_pairs)} story/summary pairs for {nn_step} data...')
        stories = '\n'.join(list(map(itemgetter(0), gen_actions.training_pairs)))
        summaries = '\n'.join(list(map(itemgetter(1), gen_actions.training_pairs)))

        mkpath(EXPORT_PATH)
        stories_dest = self.get_export_file('stories', nn_step)
        summaries_dest = self.get_export_file('summaries', nn_step)
        with open(stories_dest, 'w') as stories_file:
            stories_file.write(stories)
        with open(summaries_dest, 'w') as summaries_file:
            summaries_file.write(summaries)

    def write_training_data(self, proportion_of_test, shuffle=True):
        assert 0 <= proportion_of_test < 1
        num_test = int(proportion_of_test * len(self.training_pairs))
        if shuffle:
            print('Shuffling story/summary pairs...')
            random.shuffle(self.training_pairs)
        test_pairs = self.training_pairs[:num_test]
        train_pairs = self.training_pairs[num_test:]
        self._write_training_data(train_pairs, TRAIN)
        self._write_training_data(test_pairs, TEST)


if __name__ == '__main__':
    gen_actions = GenActions()
    gen_actions.generate_stories(story_length=5, num_stories=10)
    # gen_actions.generate_stories(story_length=4, num_stories=50)
    # gen_actions.summarise_generated_stories()
    # gen_actions.write_training_data(TEST_PROPORTION)
