import argparse
import pprint as pp
from collections import defaultdict

from helper import Helper
from parse_concept_net import ParseConceptNet

SAME_WORD_SIMILARITY = 5
SAME_WORD_POS = ['NN', 'NNS', 'NNP', 'NNPS']


class ScoreSummary:
    def __init__(self, story, summary):
        self.story = story
        self.summary = summary

        self.pcn = ParseConceptNet(False)
        self.helper = Helper()

    def score(self):
        score = self.similarity_score()

        bleu_score = self.helper.bleu_score(story, summary)
        bleu_coefficient = self.helper.count_sentences(story) / self.helper.count_sentences(summary)
        print(bleu_score, bleu_coefficient)

        return score

    def similarity_score(self):
        tokenized_story = self.helper.tokenize_text(self.story, ignore_sentence=True)
        tokenized_summary = self.helper.tokenize_text(self.summary, ignore_sentence=True)

        similar_words = defaultdict(lambda: defaultdict(lambda: 0))
        summary_similarity = 0

        for story_word, story_pos in tokenized_story:
            for summary_word, summary_pos in tokenized_summary:
                if story_pos == summary_pos and story_word == summary_word:
                    similarity = SAME_WORD_SIMILARITY if story_pos in SAME_WORD_POS else 0
                else:
                    similarity = self.pcn.compare_words(story_word, summary_word)
                    summary_similarity += similarity
                if similarity > 0:
                    similar_words[story_word][summary_word] = similarity
        pp.pprint({k: dict(v) for k, v in dict(similar_words).items()})
        return summary_similarity

    # def grammar_score(self):
    #     import grammar_check
    #     tool = grammar_check.LanguageTool('en-GB')
    #     text = 'This are bad.'
    #     matches = tool.check(text)
    #     len(matches)
    #     grammar_check.correct(text, matches)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--story', type=str, help='path to text file', required=True)
    parser.add_argument('-s', '--summary', type=str, help='path to folder with text file', required=True)
    args = parser.parse_args()
    return args.story, args.summary


if __name__ == '__main__':
    story, summary = parse_args()
    score_summary = ScoreSummary(story, summary)
    print(score_summary.score())
