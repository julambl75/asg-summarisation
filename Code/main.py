import argparse

from text_to_summary import TextToSummary


def process_args(args):
    if args.text:
        return args.text
    if args.file:
        return open(args.file).read()


def parse_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('-f', '--file', type=str, help='path to text file to summarise')
    command_group.add_argument('-t', '--text', type=str, help='text to summarise (use double quotation marks)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    text = process_args(args)
    TextToSummary(text).gen_summary()
