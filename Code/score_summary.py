import argparse
from collections import defaultdict

from ordered_set import OrderedSet

from helper import Helper


class ScoreSummary:
    def __init__(self, story, summary):
        self.story = story
        self.summary = summary

        self.helper = Helper()

    def score(self):
        tokenized_story = self.helper.tokenize_text(self.story, ignore_sentence=True)
        tokenized_summary = self.helper.tokenize_text(self.summary, ignore_sentence=True)

        similar_words = defaultdict(lambda: defaultdict(lambda: 0))

        for story_pos, story_word in tokenized_story:
            for summary_pos, summary_word in tokenized_summary:
                if story_pos == summary_pos:
                    if story_word == summary_word:
                        similarity = SAME_WORD_SIMILARITY if pos in SAME_WORD_POS else 0
                    else:
                        similarity = self.pcn.compare_words(story_word, summary_word)
                    if similarity > 0:
                        similar_words[word][other_word] = similarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--story', type=str, help='path to text file', required=True)
    parser.add_argument('-s', '--summary', type=str, help='path to folder with text file', required=True)
    args = parser.parse_args()
    return args.story, args.summary


if __name__ == '__main__':
    story, summary = parse_args()
    preprocessor = ScoreSummary(story, summary)
    preprocessor.score()
