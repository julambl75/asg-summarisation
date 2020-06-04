import functools
import itertools
import operator
import os
import re
import pprint as pp

import language_check

from parse_core_nlp import ParseCoreNLP
from score_summary import SummaryScorer

DIR = os.path.dirname(os.path.realpath(__file__)) + '/Learning/'
RULES_DIR = DIR + '/rules/'

LANGUAGE_ASG = RULES_DIR + 'general.asg'
LEARN_ACTIONS_CONSTRAINTS = RULES_DIR + 'action_constraints.asg'
LEARN_SUMMARIES_CONSTRAINTS = RULES_DIR + 'summary_constraints.asp'
LEARN_ACTIONS_BIAS = RULES_DIR + 'action_mode_bias.ilasp'

SENTENCE_RULE_IDX = 3
ACTION_RULE_SPLIT_STR = '\n\n'
SUMMARY_RULE_SPLIT_STR = '}\n'
SENTENCE_SEPARATOR = '.'

ACTION_ASG = DIR + 'action.asg'
SUMMARY_ASG = DIR + 'summary.asg'
RESULTS_FILE = DIR + 'output.txt'

DEPTH = 7
AVOID_CONSTRAINTS = "--ILASP-ss-options='-nc'"
ASG_UNSATISFIABLE = 'UNSATISFIABLE'

LEARN_ACTIONS_CMD = f'asg {ACTION_ASG} --mode=learn --depth={DEPTH} > {SUMMARY_ASG}'
GEN_SUMMARIES_CMD = f'asg {SUMMARY_ASG} --mode=run --depth={DEPTH} > {RESULTS_FILE}'
SILENCE_STD_ERR = ' 2> /dev/null'

REMOVE_SPACES_REGEX = '[^a-zA-Z0-9-]+'
FIND_BACKGROUND_REGEX = '#background *{[^}]*}'

# https://stackoverflow.com/q/35544325/
DECAPITALISE_REGEX = r'\b(?<!`)(\w+)(?!`)\b'
DECAPITALISE_QUOTE = '`'

RESTORE_PROPER_NOUNS_REGEX = (r'([a-z])([A-Z])', r'\1 \2')


class TextToSummary:
    def __init__(self, text, proper_nouns, pos_summaries=None, print_results=True):
        self.text = text
        self.proper_nouns = proper_nouns
        self.pos_summaries = pos_summaries
        self.print_results = print_results

        self.language_checker = language_check.LanguageTool('en-GB')
        self.summary_scorer = SummaryScorer()

        # Define basic ASG for learning actions for each sentence separately (reduces search space)
        self.language_asg = TextToSummary._read_file(LANGUAGE_ASG)
        self.base_action_asg = self.language_asg + TextToSummary._read_file(LEARN_ACTIONS_BIAS)

    def gen_summary(self):
        text = self._decapitalise(self.text, self.proper_nouns)
        text_parser = ParseCoreNLP(text, self.print_results)

        if self.print_results:
            print('---\nStep 1\n---')
            print('Creating context-specific ASGs and learning actions from text...')
        parsed_text = text_parser.parse_text(by_sentence=True)
        story_actions = self.learn_actions(parsed_text)

        if self.print_results:
            print('\n---\nStep 2\n---')
            print('Generating summaries, post-processing them and scoring them...')
        context_specific_asg = tuple(map(operator.itemgetter(0), parsed_text))
        scored_summaries = self.generate_summaries(story_actions, context_specific_asg)

        if self.print_results:
            pp.pprint(scored_summaries, width=200)
        return scored_summaries

    def learn_actions(self, parsed_text):
        story_actions = []
        for context_specific_asg, ilasp_constants, sentence in parsed_text:
            tokens = self._text_to_tokens(sentence)
            examples = self._gen_asg_examples(tokens)
            self._create_actions_asg()
            self._append_to_asg(ACTION_ASG, (context_specific_asg, ilasp_constants, examples))
            learned_actions = self._run_learn_actions()

            # Add indices to keep track of chronology of events
            for i, action in enumerate(learned_actions):
                learned_actions[i] = action.replace('action(', f'action({len(story_actions) + i}, ')
                if self.print_results:
                    print(learned_actions[i].strip())
            story_actions.extend(learned_actions)
        return story_actions

    def generate_summaries(self, learned_actions, context_specific_asg):
        self._create_summary_asg(learned_actions)
        self._append_to_asg(SUMMARY_ASG, context_specific_asg)
        summaries = self._gen_summary_sentences()
        summaries = self._correct_summaries(summaries)
        return summaries

    @staticmethod
    def _read_file(filename):
        return open(filename).read()

    @staticmethod
    def _decapitalise(text, proper_nouns):
        if not text:
            return ''
        for proper_noun in proper_nouns:
            text = text.replace(proper_noun, f'{DECAPITALISE_QUOTE}{proper_noun}{DECAPITALISE_QUOTE}')
        text = re.sub(DECAPITALISE_REGEX, lambda m: m.group(1).lower(), text)
        for proper_noun in proper_nouns:
            text = text.replace(f'{DECAPITALISE_QUOTE}{proper_noun}{DECAPITALISE_QUOTE}', proper_noun)
        return text

    @staticmethod
    def _text_to_tokens(text):
        text = text.split('.')
        tokens = [list(filter(lambda s: len(s) > 0, re.split(REMOVE_SPACES_REGEX, sentence))) for sentence in text]
        tokens = list(filter(lambda s: len(s) > 0, tokens))
        return [sentence + ['.'] for sentence in tokens]

    @staticmethod
    def _text_to_tokens_keep_batch(text):
        text = text.split('\n')
        token_batches = list(map(TextToSummary._text_to_tokens, text))
        tokens = [list(functools.reduce(operator.concat, token_batch, [])) for token_batch in token_batches]
        return list(filter(lambda s: len(s) > 0, tokens))

    @staticmethod
    def _gen_asg_examples(tokens_pos, tokens_neg=[]):
        return TextToSummary._gen_asg_examples_prefix(tokens_pos, '+') + \
               TextToSummary._gen_asg_examples_prefix(tokens_neg, '-')

    @staticmethod
    def _gen_asg_examples_prefix(tokens, prefix):
        return [prefix + ' [' + ', '.join(["\"{} \"".format(token) for token in sentence]) + ']' for sentence in tokens]

    def _create_actions_asg(self):
        with open(ACTION_ASG, 'w') as file:
            file.write(self.base_action_asg)

    @staticmethod
    def _append_to_asg(filename, rule_sets):
        with open(filename, 'a') as file:
            for rule_set in rule_sets:
                for rule in rule_set:
                    file.write(rule + '\n')
                file.write('\n')

    def _create_summary_asg(self, learned_actions):
        with open(SUMMARY_ASG, 'w') as file:
            file.write(self.language_asg)
        summary_constraints = TextToSummary._read_file(LEARN_SUMMARIES_CONSTRAINTS)
        with open(SUMMARY_ASG, 'r') as file:
            lang_asg = file.read()
        # Replace action detection detection constraint with summary generation constraints
        lang_asg_rules = lang_asg.split(SUMMARY_RULE_SPLIT_STR)
        lang_asg_sent_rule_lines = lang_asg_rules[SENTENCE_RULE_IDX].split('\n')
        lang_asg_sent_rule_lines = list(filter(lambda l: ':- not action' not in l, lang_asg_sent_rule_lines))
        lang_asg_sent_rule_lines.append(summary_constraints)
        lang_asg_sent_rule_lines.extend(learned_actions)

        lang_asg_rules[SENTENCE_RULE_IDX] = '\n'.join(lang_asg_sent_rule_lines) + '\n'
        lang_asg = SUMMARY_RULE_SPLIT_STR.join(lang_asg_rules)
        with open(SUMMARY_ASG, 'w') as file:
            file.write(lang_asg)

    @staticmethod
    def _update_background(filename, variables):
        background_extension = ''.join(["  {}\n".format(var) for var in variables])
        with open(filename, 'r') as file:
            filedata = file.read()
        background = re.search(FIND_BACKGROUND_REGEX, filedata).group()
        new_background = background[:-1] + background_extension + background[-1]
        filedata = filedata.replace(background, new_background)
        with open(filename, 'w') as file:
            file.write(filedata)

    def _run_learn_actions(self):
        command = LEARN_ACTIONS_CMD
        if not self.print_results:
            command += SILENCE_STD_ERR
        os.system(command)
        with open(SUMMARY_ASG, 'r') as file:
            lang_asg = file.read()
        lang_asg_rules = lang_asg.split(SUMMARY_RULE_SPLIT_STR)
        if not lang_asg or ASG_UNSATISFIABLE in lang_asg_rules[0]:
            return []
        lang_asg_sent_rule_lines = lang_asg_rules[SENTENCE_RULE_IDX].split('\n')
        return list(filter(lambda r: '  action(' in r, lang_asg_sent_rule_lines))

    def _gen_summary_sentences(self):
        command = GEN_SUMMARIES_CMD
        if not self.print_results:
            command += SILENCE_STD_ERR
        os.system(command)
        with open(RESULTS_FILE, 'r') as file:
            return list(filter(lambda l: len(l) > 0, file.read().split('\n')))

    def _get_summary_length(self):
        story_length = self.text.count(SENTENCE_SEPARATOR)
        if story_length > 4:
            return 3
        if story_length > 2:
            return 2
        return 1

    def _correct_summaries(self, summary_sentences):
        # Reverse ordering to be closer to a more chronological order of story
        summary_sentences.reverse()

        # Capitalise sentences
        summary_sentences = map(lambda s: s[0].upper() + s[1:], summary_sentences)
        # Correct grammar
        summary_sentences = map(self.language_checker.correct, summary_sentences)
        # Restore complex proper nouns
        summary_sentences = map(lambda s: re.sub(*RESTORE_PROPER_NOUNS_REGEX, s), summary_sentences)
        # Restore punctuation and fix spacing
        summary_sentences = list(map(lambda s: s.strip().replace('_', '-').replace('  ', ' '), summary_sentences))

        # Generate all order-preserving combinations of summary sentences
        summary_len = self._get_summary_length()
        summaries = {' '.join(summary) for summary in itertools.combinations(summary_sentences, summary_len)}

        # Retrieve original complex proper nouns
        proper_nouns = list(map(lambda n: re.sub(*RESTORE_PROPER_NOUNS_REGEX, n), self.proper_nouns))
        for i, proper_noun in enumerate(self.proper_nouns):
            self.text = self.text.replace(proper_noun, proper_nouns[i])

        return self.summary_scorer.asg_score(self.text, summaries, self.pos_summaries, proper_nouns)
