import argparse

from text_to_summary import TextToSummary


def process_args(args):
    summaries = None
    if args.summaries:
        summaries = open(args.summaries).read()
    if args.text:
        return args.text, summaries
    if args.file:
        return open(args.file).read(), summaries


def parse_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('-f', '--file', type=str, help='path to text file to summarise')
    command_group.add_argument('-t', '--text', type=str, help='text to summarise (use double quotation marks)')
    parser.add_argument('-s', '--summaries', type=str, help='path to file containing pos/neg summaries')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    text, summaries = process_args(args)
    TextToSummary(text, summaries).gen_summary()
