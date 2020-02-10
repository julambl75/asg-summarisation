import argparse

from text_to_summary import TextToSummary


# def process_args():
#     if len(sys.argv) > 1:
#         try:
#             text = open(sys.argv[1]).read()
#         except IOError:
#             text = sys.argv[1]
#     else:
#         print("Please pass a string or as an argument.")
#         sys.exit()
#     return text

def process_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument("-f", "--file", type=str, help="path to text file to summarise")
    command_group.add_argument("-t", "--text", type=str, help="text to summarise")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = process_args()
    print(args.echo)

    text_to_summary = TextToSummary(text).gen_summary()
