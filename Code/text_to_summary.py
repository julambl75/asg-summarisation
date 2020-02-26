import functools
import operator
import re
import os
import shutil

from parse_core_nlp import ParseCoreNLP

DIR = os.path.dirname(os.path.realpath(__file__))
ILASP_LEARN_ACTIONS = DIR + '/Learning/learn_actions.ilasp'
ILASP_LEARN_SUMMARIES = DIR + '/Learning/learn_summaries.ilasp'

INPUT_ASG = DIR + '/Learning/general.asg'
INPUT_ASG_AUGMENTED = DIR + '/Learning/general_augmented.asg'
LEARNED_ACTIONS_ASG = DIR + '/Learning/general_learned_actions.asg'
OUTPUT_ASG = DIR + '/Learning/general_output.asg'

assert INPUT_ASG != INPUT_ASG_AUGMENTED and INPUT_ASG != LEARNED_ACTIONS_ASG and INPUT_ASG != OUTPUT_ASG

DEPTH = 10
LEARN_ACTIONS_CMD = "asg '{}' --mode=learn --depth={} > '{}'".format(INPUT_ASG_AUGMENTED, DEPTH, LEARNED_ACTIONS_ASG)
LEARN_SUMMARIES_CMD = "asg '{}' --mode=learn --depth={} > '{}'".format(LEARNED_ACTIONS_ASG, DEPTH, OUTPUT_ASG)
GEN_SUMMARIES_CMD = "asg '{}' --mode=run --depth={} --ILASP-ss-options='-nc'".format(OUTPUT_ASG, DEPTH)
# TODO remove -nc flag (no constraints) if we need constraints with summary as body predicate

REMOVE_SPACES_REGEX = '[^a-zA-Z0-9]+'
FIND_BACKGROUND_REGEX = '#background *{[^}]*}'


class TextToSummary:
    def __init__(self, text, pos_summaries, neg_summaries):
        self.text = text
        self.pos_summaries = pos_summaries or ''
        self.neg_summaries = neg_summaries or ''

        self.text_parser = ParseCoreNLP(self.text, True)
        self.summaries_parser = ParseCoreNLP(self.pos_summaries + ' ' + self.neg_summaries, True)

        self.ilasp_learn_actions = open(ILASP_LEARN_ACTIONS).read().split('\n')
        self.ilasp_learn_summaries = open(ILASP_LEARN_SUMMARIES).read().split('\n')

    def gen_summary(self):
        print('---\nStep 1\n---')
        print('Generating positive examples from original text...')
        tokens = self._text_to_tokens(self.text)
        examples = self._gen_asg_examples(tokens)

        print('Parsing text to create context-specific ASG and ILASP constants...')
        context_specific_asg, ilasp_constants = self.text_parser.parse_text()

        print('Completing basic ASG with context-specific information...')
        self._copy_asg_script()
        self._append_to_asg(INPUT_ASG_AUGMENTED, (context_specific_asg, self.ilasp_learn_actions, ilasp_constants, examples))

        print('Learning actions...')
        self._run_learn_actions()

        print('\n---\nStep 2\n---')
        print('Generating positive and negative examples from reference summaries...')
        pos_tokens = self._text_to_tokens_keep_batch(self.pos_summaries)
        neg_tokens = self._text_to_tokens_keep_batch(self.neg_summaries)
        examples = self._gen_asg_examples(pos_tokens, neg_tokens)

        print('Parsing summaries to create context-specific ASG and ILASP constants...')
        context_specific_asg, ilasp_variables = self.summaries_parser.parse_text(True)

        print('Completing basic ASG with context-specific information...')
        self._update_constraints(LEARNED_ACTIONS_ASG, 'not action', 'not summary')
        self._append_to_asg(LEARNED_ACTIONS_ASG, (context_specific_asg, self.ilasp_learn_summaries, examples))
        self._update_background(LEARNED_ACTIONS_ASG, ilasp_variables)

        print('Learning summaries...')
        self._run_learn_summaries()

        print('Listing learned summaries...')
        self._run_print_summaries()

    @staticmethod
    def _text_to_tokens(text):
        text = text.lower().split('.')  # TODO allow other types of punctuation (ex: !)
        tokens = [list(filter(lambda s: len(s) > 0, re.split(REMOVE_SPACES_REGEX, sentence))) for sentence in text]
        tokens = list(filter(lambda s: len(s) > 0, tokens))
        return [sentence + ['.'] for sentence in tokens]

    @staticmethod
    def _text_to_tokens_keep_batch(text):
        text = text.lower().split('\n')
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

    @staticmethod
    def _copy_asg_script():
        shutil.copyfile(INPUT_ASG, INPUT_ASG_AUGMENTED)  # Avoids overriding original ASG file

    @staticmethod
    def _append_to_asg(filename, rule_sets):
        with open(filename, 'a') as file:
            for rule_set in rule_sets:
                for rule in rule_set:
                    file.write(rule + '\n')
                file.write('\n')

    @staticmethod
    def _update_constraints(filename, original_constraint, new_constraint):
        with open(filename, 'r') as file:
            filedata = file.read()
        filedata = filedata.replace(original_constraint, new_constraint)
        with open(filename, 'w') as file:
            file.write(filedata)

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

    @staticmethod
    def _run_learn_summaries():
        os.system(LEARN_SUMMARIES_CMD)

    @staticmethod
    def _run_print_summaries():
        os.system(GEN_SUMMARIES_CMD)
