import argparse

from preprocessor import Preprocessor
from text_to_summary import TextToSummary

FORMAT = 'txt'
POS_SUMMARIES_EXT = '_summaries_pos'
NEWLINE = '\n'


def process_args(args):
    pos = None
    if args.pos_summaries:
        pos = open(args.pos_summaries).read()
    if args.text:
        return args.text, pos
    if args.file:
        return open(args.file).read(), pos
    if args.all_files:
        path = '{}/{}'.format(args.all_files, args.all_files)
        text = open('{}.{}'.format(path, FORMAT)).read()
        try:
            pos = open('{}{}.{}'.format(path, POS_SUMMARIES_EXT, FORMAT)).read().split(NEWLINE)
        except IOError:
            pos = ''
        return text, pos


def parse_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('-f', '--file', type=str, help='path to text file to summarise')
    command_group.add_argument('-a', '--all_files', type=str, help='path to folder with text file and summaries')
    command_group.add_argument('-t', '--text', type=str, help='text to summarise (use double quotation marks)')
    parser.add_argument('-p', '--pos_summaries', type=str, help='path to file containing positive summaries')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    story, pos_summaries = process_args(args)

    print('Preprocessing text...')
    preprocessor = Preprocessor(story, print_results=False)
    homogenized_story, proper_nouns = preprocessor.preprocess()

    text_to_summary = TextToSummary(homogenized_story, proper_nouns, pos_summaries)
    text_to_summary.gen_summary()
