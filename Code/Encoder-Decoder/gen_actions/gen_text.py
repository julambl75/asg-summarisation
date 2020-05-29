import csv
import os
import random
import time
from distutils.dir_util import mkpath
from operator import itemgetter

import language_check
from datamuse import datamuse
from pattern.en import SG, PL, PRESENT

from preprocessor import Preprocessor
from query_pattern import QueryPattern
from score_summary import SummaryScorer
from text_to_summary import TextToSummary
from helper import Helper

PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PATH)

EXPORT_PATH = f'{PARENT_DIR}/data'
NAMES_FILE = f'{PARENT_DIR}/words/names.txt'
WORDS_FILE = f'{PARENT_DIR}/words/words.csv'

NOUN_FILTER = 'noun_filter'
VERB_FILTER = 'verb_filter'
ADJECTIVE_FILTER = 'adjective_filter'
DATAMUSE_MAX_NOUNS = 10
DATAMUSE_MAX_VERBS = 20
DATAMUSE_MAX_ADJECTIVES = 1

SUBJECT_TYPE_PROPER_NOUN = 'proper_noun'
SUBJECT_TYPE_PRONOUN = 'pronoun'

DETERMINERS = ['a', 'the']
CONJUNCT_KEYWORD = 'and'
PRONOUNS_SUBJECT = [('I', 1, SG), ('you', 2, SG), ('he', 3, SG), ('she', 3, SG), ('it', 3, SG),
                    ('we', 1, PL), ('you', 2, PL), ('they', 3, PL)]
SUBJECT_TO_OBJECT = {'I': 'me', 'he': 'him', 'she': 'her', 'we': 'us', 'they': 'them'}
OBJECT_TO_SUBJECT = {v: k for k, v in SUBJECT_TO_OBJECT.items()}
DEFAULT_VERB = 'be'
PUNCTUATION = '.'

PROB_PROPER_NOUN = 0
PROB_PRONOUN = 0
PROB_LAST_NOUN = 0.25

CONJUGATION_INDIVIDUAL = (3, SG)
CONJUGATION_GROUP = (3, PL)

EMPTY_TOKEN = '0'
VERB_TOKEN = 'verb'
SUBJECT_TOKEN = 'subject'
OBJECT_TOKEN = 'object'
CONJUNCT_TOKEN = 'conjunct'

PAST_TENSE = 'past'
VERB_PREDICATE = 'verb'
NOUN_PREDICATE = 'noun'
DETERMINER_PREDICATE = 'det'
ADJECTIVE_PREDICATE = 'adj_or_adv'

COMMON_NOUN_POS = 'nn'
PROPER_NOUN_POS = 'nnp'
PRONOUN_POS = 'prp'
DETERMINER_POS = 'dt'
ADJECTIVE_POS = 'jj'
PAST_TENSE_POS = 'vbd'

TRAIN = 'train'
TEST = 'test'
EVAL = 'val'

PRINT_EVERY_ITERS = 5
PRINT_EVERY_SECONDS = 150
TEST_PROPORTION = 0.1
EVAL_NUM = 5


class GenActions:
    def __init__(self):
        self.words = self._read_words()
        self.names = self._read_names()

        self.verbs = self._get_words_of_type('v')
        self.nouns = self._get_words_of_type('n')
        self.adjectives = self._get_words_of_type('j')

        self.proper_nouns = set()
        self.story_actions = []
        self.story_leaf_nodes = []
        self.stories = []
        self.training_pairs = []

        self.query_pattern = QueryPattern()
        self.datamuse_api = datamuse.Datamuse()
        self.helper = Helper()
        self.language_checker = language_check.LanguageTool('en-GB')
        self.summary_scorer = SummaryScorer()

        self._reset_for_new_story()
        self._reset_for_new_lexical_field()

    @staticmethod
    def _read_names():
        with open(NAMES_FILE, encoding='utf-8-sig') as names_file:
            return names_file.read().strip().split('\n')

    @staticmethod
    def _read_words():
        with open(WORDS_FILE) as words_csv:
            reader = csv.reader(words_csv, delimiter=',')
            return [tuple(row) for row in reader]

    def _get_words_of_type(self, word_type):
        return list(map(itemgetter(0), filter(lambda e: e[1] == word_type, self.words)))

    @staticmethod
    def _list_to_conjunct(tokens):
        if len(tokens) == 0:
            return EMPTY_TOKEN
        elif len(tokens) == 1:
            return tokens[0]
        return f'{CONJUNCT_TOKEN}({", ".join(tokens)})'

    def _reset_for_new_lexical_field(self):
        self.topic, self.lexical_common_nouns = self._get_lexical_common_nouns()
        self.lexical_verbs = self._get_lexical_verbs()

        self.topics = {self.topic}
        self.subject_type = None
        self.last_pronoun = None
        self.last_name = None

    def _reset_for_new_story(self):
        self.leaf_nodes = set()
        self.curr_story_tokens = []

    def _extract_from_datamuse(self, response, filter_type):
        words = [x['word'] for x in response if 'word' in x.keys()]
        if filter_type == NOUN_FILTER:
            words = [w for w in words if w in self.nouns or self.query_pattern.get_singular_noun(w) in self.nouns]
        elif filter_type == VERB_FILTER:
            words = [w for w in words if w in self.verbs]
        elif filter_type == ADJECTIVE_FILTER:
            words = [w for w in words if w in self.adjectives]
        return words

    def _get_lexical_common_nouns(self):
        # Try to get synonyms
        topic = random.choice(self.nouns)
        nouns = self._extract_from_datamuse(self.datamuse_api.words(rel_syn=topic, max=DATAMUSE_MAX_NOUNS), NOUN_FILTER)
        if not nouns:
            # Try to get hypernyms
            nouns = self._extract_from_datamuse(self.datamuse_api.words(rel_spc=topic, max=DATAMUSE_MAX_NOUNS), NOUN_FILTER)
        if not nouns:
            # Try to get hyponyms
            nouns = self._extract_from_datamuse(self.datamuse_api.words(rel_gen=topic, max=DATAMUSE_MAX_NOUNS), NOUN_FILTER)
        if not nouns:
            return self._get_lexical_common_nouns()
        return topic, nouns

    def _get_lexical_verbs(self):
        verbs = self._extract_from_datamuse(self.datamuse_api.words(topics=self.topic, max=DATAMUSE_MAX_VERBS), VERB_FILTER)
        return verbs if verbs else [DEFAULT_VERB]

    def _get_random_pronoun(self, token_type):
        if self.last_pronoun:
            noun, person, number = self.last_pronoun
        else:
            noun, person, number = random.choice(PRONOUNS_SUBJECT)
            self.create_asg_leaf(PRONOUN_POS, noun, NOUN_PREDICATE)
            self.last_pronoun = (noun, person, number)
        if token_type == OBJECT_TOKEN and noun in SUBJECT_TO_OBJECT.keys():
            noun = SUBJECT_TO_OBJECT[noun]
        elif token_type == SUBJECT_TOKEN and noun in OBJECT_TO_SUBJECT.keys():
            noun = OBJECT_TO_SUBJECT[noun]
        return noun, person, number

    def _get_random_proper_noun(self):
        if self.last_name:
            noun, person, number = self.last_name
        else:
            noun = random.choice(self.names)
            person, number = CONJUGATION_INDIVIDUAL
            self.proper_nouns.add(noun)
            self.create_asg_leaf(PROPER_NOUN_POS, noun, NOUN_PREDICATE)
            self.last_name = (noun, person, number)
        return noun, person, number

    def _get_random_common_noun(self, token_type):
        if token_type == OBJECT_TOKEN:
            prev_verb = self.curr_story_tokens[-1]
            prev_noun = self.curr_story_tokens[-2] if self.curr_story_tokens[-2] in self.nouns else None

            if self.helper.random_bool():
                # Get holonyms of subject appearing after verb in lexical field of topic
                response = self.datamuse_api.words(rel_com=prev_noun, topics=self.topics, lc=prev_verb, max=DATAMUSE_MAX_NOUNS)
            else:
                # Get meronyms of subject appearing after verb in lexical field of topic
                response = self.datamuse_api.words(rel_par=prev_noun, topics=self.topics, lc=prev_verb, max=DATAMUSE_MAX_NOUNS)
            nouns = self._extract_from_datamuse(response, NOUN_FILTER)
            noun = random.choice(nouns) if nouns else None
        else:
            noun = random.choice(self.lexical_common_nouns)

        # Get popular adjectives modified by chosen noun
        adjectives = self._extract_from_datamuse(self.datamuse_api.words(rel_jjb=noun, max=1), ADJECTIVE_FILTER)
        adjective = random.choice(adjectives) if adjectives else None

        if not noun and (token_type == SUBJECT_TOKEN or not adjective):
            noun = random.choice(self.nouns)
        if noun and self.query_pattern.is_plural_noun(noun):
            determiner = EMPTY_TOKEN
            person, number = CONJUGATION_GROUP
        else:
            determiner = random.choice(DETERMINERS) if noun else None
            person, number = CONJUGATION_INDIVIDUAL

        if noun:
            self.create_asg_leaf(COMMON_NOUN_POS, noun, NOUN_PREDICATE)
            self.topics.add(noun)
        if determiner:
            self.create_asg_leaf(DETERMINER_POS, determiner, DETERMINER_PREDICATE)
        if adjective:
            self.create_asg_leaf(ADJECTIVE_POS, adjective, ADJECTIVE_PREDICATE)
        return noun or EMPTY_TOKEN, determiner or EMPTY_TOKEN, adjective or EMPTY_TOKEN, person, number

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

    def get_random_subject_object(self, token_type):
        determiner = adjective_part = EMPTY_TOKEN

        # Pick random subject/object type, without using the same word for both subject and object
        if random.random() < PROB_PROPER_NOUN and self.subject_type != SUBJECT_TYPE_PROPER_NOUN:
            noun, person, number = self._get_random_proper_noun()
            self.subject_type = SUBJECT_TYPE_PROPER_NOUN if token_type == SUBJECT_TOKEN else None
        elif random.random() < PROB_PRONOUN and self.subject_type != SUBJECT_TYPE_PRONOUN:
            noun, person, number = self._get_random_pronoun(token_type)
            self.subject_type = SUBJECT_TYPE_PRONOUN if token_type == SUBJECT_TOKEN else None
        else:
            noun, determiner, adjective_part, person, number = self._get_random_common_noun(token_type)
            self.subject_type = None

        self._subject_object_to_story_tokens(noun, determiner, adjective_part)
        token = f'{token_type}({noun}, {determiner}, {adjective_part})'
        if token_type == SUBJECT_TOKEN:
            return token, noun, adjective_part, person, number
        return token

    def get_random_verb(self, person, number):
        verb = random.choice(self.lexical_verbs)
        tense = PAST_TENSE
        conjugated = self.query_pattern.conjugate(verb, person=person, tense=tense, number=number)
        self.create_asg_leaf(PAST_TENSE_POS, conjugated, VERB_PREDICATE, verb, tense)
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
        verb = self.get_random_verb(person, number)
        object = self.get_random_subject_object(OBJECT_TOKEN)
        self.curr_story_tokens.append(PUNCTUATION)
        return f'action({index}, {verb}, {subject}, {object}).'.replace('-', '_').lower()

    def format_story(self):
        joined_tokens = ' '.join(self.curr_story_tokens)
        return self.language_checker.correct(joined_tokens)

    def generate_stories(self, story_length, num_stories, irrelevant_sentence=False):
        last_print_time = start = time.time()

        for i in range(num_stories):
            time_since_print = time.time() - last_print_time
            if (i % PRINT_EVERY_ITERS == 0 or time_since_print > PRINT_EVERY_SECONDS) and i > 0:
                last_print_time = time.time()
                time_ind = self.helper.time_since(start, i / num_stories)
                print(f'{time_ind} - [{i}/{num_stories}]: Generating stories of length {story_length}...')

            story_actions = []
            for action_idx in range(story_length):
                story_actions.append(self.generate_action(action_idx))
            self._reset_for_new_lexical_field()

            self.story_actions.append(story_actions)
            self.story_leaf_nodes.append(sorted(self.leaf_nodes))
            self.stories.append(self.format_story())
            self._reset_for_new_story()
        print(f'[{num_stories}/{num_stories}]: Generated stories of length {story_length}...')

    def summarise_generated_stories(self):
        last_print_time = start = time.time()
        num_stories = len(self.story_actions)

        for action_set, leaf_node_set, story in zip(self.story_actions, self.story_leaf_nodes, self.stories):
            time_since_print = time.time() - last_print_time
            num_summaries = len(self.training_pairs)
            if (num_summaries % PRINT_EVERY_ITERS == 0 or time_since_print > PRINT_EVERY_SECONDS) and num_summaries > 0:
                last_print_time = time.time()
                time_ind = self.helper.time_since(start, num_summaries / num_stories)
                print(f'{time_ind} - [{len(self.training_pairs)}/{num_stories}]: Summarising generated stories...')

            if action_set and leaf_node_set:
                text_to_summary = TextToSummary(story, self.proper_nouns, print_results=False)
                summaries = text_to_summary.generate_summaries(action_set, (leaf_node_set,))
            else:
                preprocessor = Preprocessor(story, print_results=False)
                story, proper_nouns = preprocessor.preprocess()
                text_to_summary = TextToSummary(story, proper_nouns, print_results=False)
                summaries = text_to_summary.gen_summary()
            if summaries:
                self.training_pairs.append((story, summaries[0][0]))
        print(f'[{num_stories}/{num_stories}]: Summarised generated stories...')

    @staticmethod
    def get_export_file(data_type, nn_step):
        return f'{EXPORT_PATH}/{data_type}_{nn_step}.txt'

    def _write_training_data(self, story_summary_pairs, nn_step):
        print(f'Writing {len(story_summary_pairs)} story/summary pairs for {nn_step} data...')
        stories = '\n'.join(list(map(itemgetter(0), story_summary_pairs))).replace('.', ' .').lower()
        summaries = '\n'.join(list(map(itemgetter(1), story_summary_pairs))).replace('.', ' .').lower()

        mkpath(EXPORT_PATH)
        stories_dest = self.get_export_file('stories', nn_step)
        summaries_dest = self.get_export_file('summaries', nn_step)
        with open(stories_dest, 'a') as stories_file:
            stories_file.write(stories)
        with open(summaries_dest, 'a') as summaries_file:
            summaries_file.write(summaries)

    def write_training_data(self, proportion_of_test, num_eval, shuffle=True):
        assert 0 <= proportion_of_test < 1
        assert 0 <= num_eval < len(self.training_pairs) / 10

        num_not_eval = len(self.training_pairs) - num_eval
        num_test = int(proportion_of_test * num_not_eval)
        if shuffle:
            print('Shuffling story/summary pairs...')
            random.shuffle(self.training_pairs)
        train_pairs = self.training_pairs[num_eval+num_test:]
        test_pairs = self.training_pairs[num_eval:num_eval+num_test]
        eval_pairs = self.training_pairs[:num_eval]
        self._write_training_data(train_pairs, TRAIN)
        self._write_training_data(test_pairs, TEST)
        self._write_training_data(eval_pairs, EVAL)


if __name__ == '__main__':
    gen_actions = GenActions()
    gen_actions.generate_stories(story_length=4, num_stories=2000, irrelevant_sentence=True)
    gen_actions.summarise_generated_stories()
    gen_actions.write_training_data(TEST_PROPORTION, EVAL_NUM)
