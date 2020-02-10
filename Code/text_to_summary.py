import sys
import re
import os
import shutil

DIR = os.path.dirname(os.path.realpath(__file__))
PARSE_WORDS_SCRIPT = DIR + '/parse_core_nlp.py'
INPUT_ASG = DIR + '/Learning/general.asg'
INPUT_ASG_TMP = DIR + '/Learning/general_tmp.asg'
OUTPUT_ASG = DIR + '/Learning/general_learned.asg'

assert INPUT_ASG != INPUT_ASG_TMP
assert INPUT_ASG != OUTPUT_ASG

PARSE_WORDS_CMD = "python3 {} {} --no_tree".format(PARSE_WORDS_SCRIPT)

DEPTH = 10
LEARN_CMD = "asg '{}' --mode=learn --depth={} > '{}'".format(INPUT_ASG_TMP, DEPTH, OUTPUT_ASG)
GEN_SUMMARIES_CMD = "asg '{}' --mode=run --depth={}".format(OUTPUT_ASG, DEPTH, OUTPUT_ASG)

REMOVE_SPACES_REGEX = '[^a-zA-Z0-9]+'

# Process whole documents
if len(sys.argv) > 1:
    try:
        text = open(sys.argv[1]).read()
    except IOError:
        text = sys.argv[1]
else:
    print("Please pass a story in string form as an argument.")
    sys.exit()

# TODO Generate text-specific ASG leaf nodes and ILASP constants

# Format text for ASG
text = text.lower().split('.') # TODO allow other types of punctuation
tokens = [list(filter(lambda s: len(s) > 0, re.split(REMOVE_SPACES_REGEX, sentence))) for sentence in text]
tokens = list(filter(lambda s: len(s) > 0, tokens))
tokens = [sentence + ['.'] for sentence in tokens] # TODO remove readability punctuation in examples

# Generate positive examples
pos_examples = ['+ [' + ', '.join(["\"{} \"".format(token) for token in sentence]) + ']' for sentence in tokens]

# Copy ASG file to avoid overriding
shutil.copyfile(INPUT_ASG, INPUT_ASG_TMP)

# Append positive example (story) to temporary ASG file
tmp = open(INPUT_ASG_TMP, 'a')
[tmp.write(example + '\n') for example in pos_examples]
tmp.close()

# Generate summaries
os.system(LEARN_CMD)
os.system(GEN_SUMMARIES_CMD)
