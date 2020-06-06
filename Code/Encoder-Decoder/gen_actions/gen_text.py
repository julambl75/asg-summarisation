import csv
import os
import random
import time
from distutils.dir_util import mkpath
from operator import itemgetter

from datamuse import datamuse
from pattern.en import SG

from query_pattern import QueryPattern
from score_summary import SummaryScorer
from text_to_summary import TextToSummary
from helper import Helper

PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PATH)

EXPORT_PATH = f'{PARENT_DIR}/data'
WORDS_FILE = f'{PARENT_DIR}/words/words.csv'

NOUN_FILTER = 'noun_filter'
VERB_FILTER = 'verb_filter'
ADJECTIVE_FILTER = 'adjective_filter'
DATAMUSE_MAX_NOUNS = 10
DATAMUSE_MAX_VERBS = 20

DETERMINERS = ['a', 'the']
DESCRIPTIVE_PREP = 'it'
DEFAULT_VERB = 'be'
PUNCTUATION = '.'

EMPTY_TOKEN = '0'
VERB_TOKEN = 'verb'
SUBJECT_TOKEN = 'subject'
OBJECT_TOKEN = 'object'

PAST_TENSE = 'past'
VERB_PREDICATE = 'verb'
NOUN_PREDICATE = 'noun'
DETERMINER_PREDICATE = 'det'
ADJECTIVE_PREDICATE = 'adj_or_adv'

NOUN_POS = 'nn'
PREPOSITION_POS = 'prp'
DETERMINER_POS = 'dt'
ADJECTIVE_POS = 'jj'
PAST_TENSE_POS = 'vbd'

# See README
CONJUNCTIVE_SUMMARY = 'conjunctive'
DESCRIPTIVE_SUMMARY = 'descriptive'

TRAIN = 'train'
TEST = 'test'
VALID = 'val'

PRINT_EVERY_ITERS = 5
PRINT_EVERY_SECONDS = 150
VALID_PROPORTION = 0.1
TEST_NUM = 10


class GenActions:
    def __init__(self):
        self.words = self._read_words()
        self.verbs = self._get_words_of_type('v')
        self.nouns = self._get_words_of_type('n')
        self.adjectives = self._get_words_of_type('j')

        self.story_actions = []
        self.story_leaf_nodes = []
        self.stories = []
        self.training_pairs = []

        self.query_pattern = QueryPattern()
        self.datamuse_api = datamuse.Datamuse()
        self.helper = Helper()
        self.summary_scorer = SummaryScorer()

    @staticmethod
    def _read_words():
        with open(WORDS_FILE) as words_csv:
            reader = csv.reader(words_csv, delimiter=',')
            return [tuple(row) for row in reader]

    def _get_words_of_type(self, word_type):
        return list(map(itemgetter(0), filter(lambda e: e[1] == word_type, self.words)))

    def _reset_for_new_story(self):
        self.topic = random.choice(self.nouns)
        self.lexical_verb = self._get_lexical_verb()
        self.used_nouns = {self.topic}

        self.leaf_nodes = set()
        self.curr_story_tokens = []

        self.story_subject, self.start_sent_tokens = self._create_subject_object(SUBJECT_TOKEN)
        self.story_verb, verb_tokens = self._conjugate_verb(self.lexical_verb)
        self.start_sent_tokens.append(verb_tokens)

    def _extract_from_datamuse(self, response, filter_type):
        words = [x['word'] for x in response if 'word' in x.keys()]
        if filter_type == NOUN_FILTER:
            words = [w for w in words if w in self.nouns or self.query_pattern.get_singular_noun(w) in self.nouns]
        elif filter_type == VERB_FILTER:
            words = [w for w in words if w in self.verbs]
        elif filter_type == ADJECTIVE_FILTER:
            words = [w for w in words if w in self.adjectives]
        return words

    def _get_lexical_verb(self):
        verbs = self._extract_from_datamuse(self.datamuse_api.words(topics=self.topic, max=DATAMUSE_MAX_VERBS), VERB_FILTER)
        return random.choice(verbs) if verbs else DEFAULT_VERB

    def _find_popular_adjective(self, noun, must_return=False):
        adjectives = self._extract_from_datamuse(self.datamuse_api.words(rel_jjb=noun, max=1), ADJECTIVE_FILTER)
        if adjectives:
            adjective = adjectives[0]
        elif must_return:
            adjective = random.choice(self.adjectives)
        else:
            return None
        self._create_asg_leaf(ADJECTIVE_POS, adjective, ADJECTIVE_PREDICATE)
        return adjective

    def _get_random_subject_object(self, token_type):
        if token_type == OBJECT_TOKEN:
            prev_verb, prev_noun = set(self.curr_story_tokens[-2:])
            if self.helper.random_bool():
                # Get holonyms of subject appearing after verb in lexical field of topic
                response = self.datamuse_api.words(rel_com=prev_noun, topics=self.used_nouns, lc=prev_verb, max=DATAMUSE_MAX_NOUNS)
            else:
                # Get meronyms of subject appearing after verb in lexical field of topic
                response = self.datamuse_api.words(rel_par=prev_noun, topics=self.used_nouns, lc=prev_verb, max=DATAMUSE_MAX_NOUNS)
            nouns = self._extract_from_datamuse(response, NOUN_FILTER)
            noun = random.choice(nouns) if nouns else None
        else:
            noun = self.topic

        determiner = None
        adjective = self._find_popular_adjective(noun)

        if not noun and not adjective:
            noun = random.choice(self.nouns)
        if noun:
            determiner = random.choice(DETERMINERS)
            if self.query_pattern.is_plural_noun(noun):
                self.query_pattern.get_singular_noun(noun)
            self.used_nouns.add(noun)

            self._create_asg_leaf(NOUN_POS, noun, NOUN_PREDICATE)
            self._create_asg_leaf(DETERMINER_POS, determiner, DETERMINER_PREDICATE)
        return noun, determiner, adjective

    def _create_subject_object(self, token_type, noun=None, determiner=None, adjective=None):
        if not noun:
            # We need to pick a random noun
            noun, determiner, adjective = self._get_random_subject_object(token_type)
        elif not adjective and noun != DESCRIPTIVE_PREP:
            # A noun has already been chosen and if it's not the preposition 'it' we look for a relevant adjective
            adjective = self._find_popular_adjective(noun, must_return=True)
            determiner = random.choice(DETERMINERS)
        rule = f'{token_type}({noun or EMPTY_TOKEN}, {determiner or EMPTY_TOKEN}, {adjective or EMPTY_TOKEN})'
        words = list(filter(lambda t: t, (determiner, adjective, noun)))

        # Return subject story tokens so they can be added at start of each sentence
        if token_type == SUBJECT_TOKEN:
            return rule, words
        self.curr_story_tokens.extend(words)
        return rule

    def _conjugate_verb(self, verb=DEFAULT_VERB):
        tense = PAST_TENSE
        conjugated = self.query_pattern.conjugate(verb, person=3, tense=PAST_TENSE, number=SG)
        return self._create_asg_leaf(PAST_TENSE_POS, conjugated, VERB_PREDICATE, verb, tense), conjugated

    def _create_asg_leaf(self, pos_tag, value, predicate, verb_name=None, verb_form=None):
        if predicate == VERB_PREDICATE:
            lemma = f'{verb_name}, {verb_form}'
        else:
            lemma = value.lower().replace('-', '_')
        leaf_rule = f'{predicate}({lemma})'
        leaf_node = f'{pos_tag} -> "{value} " {{ {leaf_rule}. }}'
        self.leaf_nodes.add(leaf_node)
        return leaf_rule

    def _generate_action(self, index, subject, verb, object=None, use_topic_subject_and_verb=True):
        if use_topic_subject_and_verb:
            self.curr_story_tokens.extend(self.start_sent_tokens)
        if not object:
            object = self._create_subject_object(OBJECT_TOKEN)
        self.curr_story_tokens.append(PUNCTUATION)
        return f'action({index}, {verb}, {subject}, {object}).'.replace('-', '_').lower()

    def _generate_descriptive_action(self, index):
        assert index > 0
        described_noun = self.curr_story_tokens[-2]

        subject, subject_tokens = self._create_subject_object(SUBJECT_TOKEN, noun=DESCRIPTIVE_PREP)
        self._create_asg_leaf(NOUN_POS, DESCRIPTIVE_PREP, NOUN_PREDICATE)
        self.curr_story_tokens.extend(subject_tokens)

        verb, verb_token = self._conjugate_verb(DEFAULT_VERB)
        self.curr_story_tokens.append(verb_token)

        object = self._create_subject_object(OBJECT_TOKEN, noun=described_noun)
        action = self._generate_action(index, subject, verb, object, use_topic_subject_and_verb=False)

        return action

    def generate_story(self, story_type, story_length=None):
        self._reset_for_new_story()

        if story_type == DESCRIPTIVE_SUMMARY:
            main_action = self._generate_action(0, self.story_subject, self.story_verb)
            descriptive_action = self._generate_descriptive_action(1)

            story_actions = [main_action, descriptive_action]
        else:
            assert story_length
            story_actions = []

            for action_idx in range(story_length):
                action = self._generate_action(action_idx, self.story_subject, self.story_verb)
                story_actions.append(action)

        self.story_actions.append(story_actions)
        self.story_leaf_nodes.append(sorted(self.leaf_nodes))
        self.stories.append(' '.join(self.curr_story_tokens))

    def generate_stories(self, story_type, num_stories, story_length=None):
        status_detail = f'stories of type {story_type}'
        if story_length:
            status_detail += f' and length {story_length}'
        last_print_time = start = time.time()

        for i in range(num_stories):
            time_since_print = time.time() - last_print_time
            if (i % PRINT_EVERY_ITERS == 0 or time_since_print > PRINT_EVERY_SECONDS) and i > 0:
                last_print_time = time.time()
                time_ind = self.helper.time_since(start, i / num_stories)
                print(f'{time_ind} - [{i}/{num_stories}]: Generating {status_detail}...')
            self.generate_story(story_type, story_length)

        print(f'[{num_stories}/{num_stories}]: Generated {status_detail}...')

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

            text_to_summary = TextToSummary(story, print_results=False)
            summaries = text_to_summary.generate_summaries(action_set, (leaf_node_set,))
            if summaries:
                # Prioritise summaries which contain more information
                summary = max(list(map(itemgetter(0), summaries)), key=len)
            else:
                # Should not happen but avoids crash while generating large set of training data
                summary = ''
            self.training_pairs.append((story, summary))

        print(f'[{num_stories}/{num_stories}]: Summarised generated stories...')

    @staticmethod
    def _get_export_file(data_type, nn_step):
        return f'{EXPORT_PATH}/{data_type}_{nn_step}.txt'

    def _write_training_data(self, story_summary_pairs, nn_step):
        print(f'Writing {len(story_summary_pairs)} story/summary pairs for {nn_step} data...')
        stories = '\n'.join(list(map(itemgetter(0), story_summary_pairs))).lower()
        summaries = '\n'.join(list(map(itemgetter(1), story_summary_pairs))).replace('.', ' .').lower()

        mkpath(EXPORT_PATH)
        stories_dest = self._get_export_file('stories', nn_step)
        summaries_dest = self._get_export_file('summaries', nn_step)
        with open(stories_dest, 'w') as stories_file:
            stories_file.write(stories)
        with open(summaries_dest, 'w') as summaries_file:
            summaries_file.write(summaries)

    def write_training_data(self, proportion_of_valid, num_test, shuffle=True):
        assert 0 <= proportion_of_valid < 1
        assert 0 <= num_test < len(self.training_pairs) / 10

        num_not_test = len(self.training_pairs) - num_test
        num_valid = int(proportion_of_valid * num_not_test)
        if shuffle:
            print('Shuffling story/summary pairs...')
            random.shuffle(self.training_pairs)
        train_pairs = self.training_pairs[num_test + num_valid:]
        valid_pairs = self.training_pairs[num_test:num_test + num_valid]
        test_pairs = self.training_pairs[:num_test]
        self._write_training_data(train_pairs, TRAIN)
        self._write_training_data(valid_pairs, VALID)
        self._write_training_data(test_pairs, TEST)


if __name__ == '__main__':
    gen_actions = GenActions()
    gen_actions.generate_stories(CONJUNCTIVE_SUMMARY, num_stories=2000, story_length=3)
    gen_actions.generate_stories(DESCRIPTIVE_SUMMARY, num_stories=2000)
    gen_actions.summarise_generated_stories()
    gen_actions.write_training_data(VALID_PROPORTION, TEST_NUM)
