import argparse

from text_to_summary import TextToSummary

FORMAT = 'txt'
POS_SUMMARIES_EXT = '_summaries_pos'
NEG_SUMMARIES_EXT = '_summaries_neg'


def process_args(args):
    pos_summaries = None
    neg_summaries = None
    if args.pos_summaries:
        pos_summaries = open(args.pos_summaries).read()
    if args.neg_summaries:
        neg_summaries = open(args.neg_summaries).read()
    if args.text:
        return args.text, pos_summaries, neg_summaries
    if args.file:
        return open(args.file).read(), pos_summaries, neg_summaries
    if args.all_files:
        path = '{}/{}'.format(args.all_files, args.all_files)
        text = open('{}.{}'.format(path, FORMAT)).read()
        pos_summaries = open('{}{}.{}'.format(path, POS_SUMMARIES_EXT, FORMAT)).read()
        neg_summaries = open('{}{}.{}'.format(path, NEG_SUMMARIES_EXT, FORMAT)).read()
        return text, pos_summaries, neg_summaries


def parse_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('-f', '--file', type=str, help='path to text file to summarise')
    command_group.add_argument('-a', '--all_files', type=str, help='path to folder with text file and summaries')
    command_group.add_argument('-t', '--text', type=str, help='text to summarise (use double quotation marks)')
    parser.add_argument('-p', '--pos_summaries', type=str, help='path to file containing positive summaries')
    parser.add_argument('-n', '--neg_summaries', type=str, help='path to file containing negative summaries')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    text_to_summary = TextToSummary(*process_args(args))
    text_to_summary.gen_summary()
