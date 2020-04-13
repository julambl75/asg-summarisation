import argparse
import pprint as pp
from collections import defaultdict

import language_check

from helper import Helper
from parse_concept_net import ParseConceptNet

SAME_WORD_SIMILARITY = 5
SAME_WORD_POS = ['NN', 'NNS', 'NNP', 'NNPS']

IGNORE_GRAMMAR_ERRORS = ['MORFOLOGIK_RULE_EN_GB']
GRAMMAR_ERROR_PENALTY = 0.75


class SummaryScorer:
    def __init__(self, story, summary):
        self.story = story
        self.summary = summary

        self.pcn = ParseConceptNet(False)
        self.helper = Helper()
        self.language_checker = language_check.LanguageTool('en-GB')

    def score(self):
        similarity_score = self.similarity_score()

        bleu_score = self.helper.bleu_score(story, summary)
        bleu_score *= self.helper.count_sentences(story) / self.helper.count_sentences(summary)

        grammar_penalty = self.grammar_penalty()

        length_penalty = self.length_penalty()

        return similarity_score * bleu_score * grammar_penalty * length_penalty

    def similarity_score(self):
        tokenized_story = self.helper.tokenize_text(self.story, ignore_sentence=True)
        tokenized_summary = self.helper.tokenize_text(self.summary, ignore_sentence=True)

        similar_words = defaultdict(lambda: defaultdict(lambda: 0))
        summary_similarity = 0

        for story_word, story_pos in tokenized_story:
            for summary_word, summary_pos in tokenized_summary:
                if story_pos == summary_pos:
                    if story_word == summary_word:
                        similarity = SAME_WORD_SIMILARITY if story_pos in SAME_WORD_POS else 0
                    else:
                        similarity = self.pcn.compare_words(story_word, summary_word)
                        summary_similarity += similarity
                    if similarity > 0:
                        similar_words[story_word][summary_word] = similarity
        pp.pprint({k: dict(v) for k, v in dict(similar_words).items()})
        return summary_similarity

    # Decrease final score by 25% for every grammar error other than uncommon proper noun
    def grammar_penalty(self):
        errors = self.language_checker.check(self.summary)
        num_errors = len(list(filter(lambda m: m.ruleId not in IGNORE_GRAMMAR_ERRORS, errors)))
        return GRAMMAR_ERROR_PENALTY ** num_errors

    # Divide final score by 2 for ever sentence over limit in summary
    def length_penalty(self):
        story_length = self.helper.count_sentences(self.story)
        if story_length > 3:
            penalty = abs(self.helper.count_sentences(self.summary) - 3)
        else:
            penalty = abs(self.helper.count_sentences(self.summary) - story_length + 1)
        return 1 / (2 ** penalty)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--story', type=str, help='path to text file', required=True)
    parser.add_argument('-s', '--summary', type=str, help='path to folder with text file', required=True)
    args = parser.parse_args()
    return args.story, args.summary


if __name__ == '__main__':
    story, summary = parse_args()
    summary_scorer = SummaryScorer(story, summary)
    score = summary_scorer.score()
    print(score)
