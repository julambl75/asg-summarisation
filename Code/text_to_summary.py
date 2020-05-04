import functools
import operator
import os
import re

import language_check

from parse_core_nlp import ParseCoreNLP

DIR = os.path.dirname(os.path.realpath(__file__)) + '/Learning/'
REF_DIR = DIR + '/ref/'

LANGUAGE_ASG = REF_DIR + 'general.asg'
LEARN_ACTIONS_CONSTRAINTS = REF_DIR + 'action_constraints.asg'
LEARN_SUMMARIES_CONSTRAINTS = REF_DIR + 'summary_constraints.asp'
LEARN_ACTIONS_BIAS = REF_DIR + 'action_mode_bias.ilasp'

SENTENCE_RULE_IDX = 3
ACTION_RULE_SPLIT_STR = '\n\n'
SUMMARY_RULE_SPLIT_STR = '}\n'

ACTION_ASG = DIR + 'action.asg'
SUMMARY_ASG = DIR + 'summary.asg'
RESULTS_FILE = DIR + 'output.txt'

DEPTH = 10

# TODO command to learn summary rules using "--ILASP-ss-options='-nc'"
LEARN_ACTIONS_CMD = f'asg {ACTION_ASG} --mode=learn --depth={DEPTH} > {SUMMARY_ASG}'
GEN_SUMMARIES_CMD = f'asg {SUMMARY_ASG} --mode=run --depth={DEPTH} > {RESULTS_FILE}'

REMOVE_SPACES_REGEX = '[^a-zA-Z0-9-]+'
FIND_BACKGROUND_REGEX = '#background *{[^}]*}'


class TextToSummary:
    def __init__(self, text, pos_summaries, neg_summaries, proper_nouns):
        self.text = self._decapitalise(text, proper_nouns)
        # self.pos_summaries = self._decapitalise(pos_summaries, proper_nouns)
        # self.neg_summaries = self._decapitalise(neg_summaries, proper_nouns)

        self.text_parser = ParseCoreNLP(self.text, True)
        self.language_checker = language_check.LanguageTool('en-GB')
        # self.summaries_parser = ParseCoreNLP(self.pos_summaries + ' ' + self.neg_summaries, True)

        # Define basic ASG for learning actions for each sentence separately (reduces search space)
        self.language_asg = TextToSummary._read_file(LANGUAGE_ASG)
        self.base_action_asg = self.language_asg + TextToSummary._read_file(LEARN_ACTIONS_BIAS)

    def gen_summary(self):
        print('---\nStep 1\n---')
        print('Creating context-specific ASG and learning actions from text...')
        learned_actions = []
        parsed_text = self.text_parser.parse_text(by_sentence=True)

        for context_specific_asg, ilasp_constants, sentence in parsed_text:
            tokens = self._text_to_tokens(sentence)
            examples = self._gen_asg_examples(tokens)

            self._create_actions_asg()
            self._append_to_asg(ACTION_ASG, (context_specific_asg, ilasp_constants, examples))
            learned_actions.extend(self._run_learn_actions())

        # Add indices to keep track of chronology of events
        learned_actions = [action.replace('action(', f'action({i}, ') for i, action in enumerate(learned_actions)]
        for action in learned_actions:
            print(action.strip())

        print('\n---\nStep 2\n---')
        # print('Generating positive and negative examples from reference summaries...')
        # pos_tokens = self._text_to_tokens_keep_batch(self.pos_summaries)
        # neg_tokens = self._text_to_tokens_keep_batch(self.neg_summaries)
        # examples = self._gen_asg_examples(pos_tokens, neg_tokens)

        # print('Parsing summaries to create context-specific ASG and ILASP constants...')
        # context_specific_asg, ilasp_variables = self.summaries_parser.parse_text(True)

        print('Updating ASG constraints...')
        context_specific_asg = tuple(map(operator.itemgetter(0), parsed_text))
        self._create_summary_asg(learned_actions)
        self._append_to_asg(SUMMARY_ASG, context_specific_asg)
        # self._append_to_asg(SUMMARY_ASG, (*context_specific_asg, *ilasp_constants))
        # self._append_to_asg(SUMMARY_ASG, (context_specific_asg, examples))
        # self._update_background(SUMMARY_ASG, ilasp_variables)

        # print('Learning summaries...')
        # self._run_learn_summaries()

        print('Generating summaries...')
        summaries = self._gen_summaries()

        print('Post-processing summaries...')
        # summaries = self._correct_summaries(summaries)

        return summaries

    def learn_actions(self, parsed_text):
        pass

    def generate_summaries(self, learned_actions, context_specific_asg):
        pass

    @staticmethod
    def _read_file(filename):
        return open(filename).read()

    @staticmethod
    def _decapitalise(text, proper_nouns):
        if not text:
            return ''
        text = text.lower()
        for proper_noun in proper_nouns:
            text = text.replace(proper_noun.lower(), proper_noun)
        return text

    @staticmethod
    def _text_to_tokens(text):
        text = text.split('.')  # TODO allow other types of punctuation (ex: !)
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
            file.write(self.base_action_asg)
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

    @staticmethod
    def _run_learn_actions():
        os.system(LEARN_ACTIONS_CMD)
        with open(SUMMARY_ASG, 'r') as file:
            lang_asg = file.read()
        lang_asg_rules = lang_asg.split(SUMMARY_RULE_SPLIT_STR)
        lang_asg_sent_rule_lines = lang_asg_rules[SENTENCE_RULE_IDX].split('\n')
        return list(filter(lambda r: '  action(' in r, lang_asg_sent_rule_lines))

    @staticmethod
    def _gen_summaries():
        os.system(GEN_SUMMARIES_CMD)
        with open(RESULTS_FILE, 'r') as file:
            return list(filter(lambda l: len(l) > 0, file.read().split('\n')))

    def _correct_summaries(self, summaries):
        return set(map(lambda s: s.strip().replace('_', '-'), map(self.language_checker.correct, summaries)))
