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
    def __init__(self, text, summaries):
        self.text = text
        self.summaries = summaries
        self.language_parser = ParseCoreNLP(text, True)

    def gen_summary(self):
        print('Generating positive examples to learn narrative...')
        self._text_to_tokens()
        self._gen_pos_examples()

        print('Parsing text to create context-specific ASG and ILASP constants...')
        context_specific_asg, ilasp_constants = self.language_parser.parse_text()

        print('Completing basic ASG with context-specific information...')
        self._copy_asg_script()
        self._append_to_asg(context_specific_asg)
        self._append_to_asg(ilasp_constants)
        self._append_to_asg(self.pos_examples)

        print('Running ASG...')
        self._run_asg()

        # TODO
        print('Learning summaries...')

    def _text_to_tokens(self):
        # Format text for ASG
        text = self.text.lower().split('.')  # TODO allow other types of punctuation (ex: !)
        tokens = [list(filter(lambda s: len(s) > 0, re.split(REMOVE_SPACES_REGEX, sentence))) for sentence in text]
        tokens = list(filter(lambda s: len(s) > 0, tokens))
        self.tokens = [sentence + ['.'] for sentence in tokens]  # TODO remove readability punctuation in examples

    def _gen_pos_examples(self):
        self.pos_examples = ['+ [' + ', '.join(["\"{} \"".format(token) for token in sentence]) + ']' for sentence in self.tokens]

    @staticmethod
    def _copy_asg_script():
        shutil.copyfile(INPUT_ASG, INPUT_ASG_AUGMENTED)  # Avoids overriding original ASG file

    @staticmethod
    def _append_to_asg(rules):
        tmp = open(INPUT_ASG_AUGMENTED, 'a')
        [tmp.write(example + '\n') for example in rules + ['']]
        tmp.close()

    @staticmethod
    def _run_asg():
        os.system(LEARN_ACTIONS_CMD)
        os.system(GEN_SUMMARIES_CMD)
