import re
import os
import shutil

from parse_core_nlp import ParseCoreNLP

DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_ASG = DIR + '/Learning/general.asg'
INPUT_ASG_AUGMENTED = DIR + '/Learning/general_augmented.asg'
LEARNED_ACTIONS_ASG = DIR + '/Learning/general_learned_actions.asg'
OUTPUT_ASG = DIR + '/Learning/general_output.asg'

assert INPUT_ASG != INPUT_ASG_AUGMENTED and INPUT_ASG != LEARNED_ACTIONS_ASG and INPUT_ASG != OUTPUT_ASG

DEPTH = 10
LEARN_ACTIONS_CMD = "asg '{}' --mode=learn --depth={} > '{}'".format(INPUT_ASG_AUGMENTED, DEPTH, LEARNED_ACTIONS_ASG)
LEARN_SUMMARIES_CMD = "asg '{}' --mode=learn --depth={} > '{}'".format(LEARNED_ACTIONS_ASG, DEPTH, OUTPUT_ASG)
GEN_SUMMARIES_CMD = "asg '{}' --mode=run --depth={}".format(OUTPUT_ASG, DEPTH)
# TODO call new commands

REMOVE_SPACES_REGEX = '[^a-zA-Z0-9]+'


class TextToSummary:
    def __init__(self, text, pos_summaries, neg_summaries):
        self.text = text
        self.pos_summaries = pos_summaries or ''
        self.neg_summaries = neg_summaries or ''
        self.text_parser = ParseCoreNLP(self.text, True)
        self.summaries_parser = ParseCoreNLP(self.pos_summaries + ' ' + self.neg_summaries, True)

    def gen_summary(self):
        print('---\nStep 1\n---')
        print('Generating positive examples from original text...')
        text_tokens = self._text_to_tokens(self.text)
        text_examples = self._gen_asg_examples(text_tokens)

        print('Parsing text to create context-specific ASG and ILASP constants...')
        context_specific_asg, ilasp_constants = self.text_parser.parse_text()

        print('Completing basic ASG with context-specific information...')
        self._copy_asg_script()
        self._append_to_asg(INPUT_ASG_AUGMENTED, context_specific_asg)
        self._append_to_asg(INPUT_ASG_AUGMENTED, ilasp_constants)
        self._append_to_asg(INPUT_ASG_AUGMENTED, text_examples)

        print('Learning actions...')
        self._run_learn_actions()

        print('\n---\nStep 2\n---')
        print('Generating positive and negative examples from reference summaries...')
        pos_summary_tokens = self._text_to_tokens(self.pos_summaries)
        neg_summary_tokens = self._text_to_tokens(self.neg_summaries)
        summary_examples = self._gen_asg_examples(pos_summary_tokens, neg_summary_tokens)

        print('Parsing summaries to create context-specific ASG and ILASP constants...')
        context_specific_asg, ilasp_constants = self.summaries_parser.parse_text()

        print('Completing basic ASG with context-specific information...')
        self._append_to_asg(LEARNED_ACTIONS_ASG, context_specific_asg)
        self._append_to_asg(LEARNED_ACTIONS_ASG, ilasp_constants)
        self._append_to_asg(LEARNED_ACTIONS_ASG, summary_examples)

        print('Learning summaries...')
        self._run_learn_summaries()
        self._run_print_summaries()

    @staticmethod
    def _text_to_tokens(text):
        # Format text for ASG
        text = text.lower().split('.')  # TODO allow other types of punctuation (ex: !)
        tokens = [list(filter(lambda s: len(s) > 0, re.split(REMOVE_SPACES_REGEX, sentence))) for sentence in text]
        tokens = list(filter(lambda s: len(s) > 0, tokens))
        return [sentence + ['.'] for sentence in tokens]  # TODO remove readability punctuation in examples

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
    def _append_to_asg(file, rules):
        tmp = open(file, 'a')
        [tmp.write(example + '\n') for example in rules + ['']]
        tmp.close()

    @staticmethod
    def _run_learn_actions():
        os.system(LEARN_ACTIONS_CMD)

    @staticmethod
    def _run_learn_summaries():
        os.system(LEARN_SUMMARIES_CMD)

    @staticmethod
    def _run_print_summaries():
        os.system(GEN_SUMMARIES_CMD)
