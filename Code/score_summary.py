import argparse
import pprint as pp
from collections import defaultdict

import language_check

from helper import Helper
from parse_concept_net import ParseConceptNet

BATCH_STORY_PREFIX = '> '
BATCH_SUMMARY_PREFIX = '< '
EOS_POSTFIX = '<EOS>'

SAME_WORD_SIMILARITY = 5
SAME_WORD_POS = ['NN', 'NNS', 'NNP', 'NNPS']

IGNORE_GRAMMAR_ERRORS = ['MORFOLOGIK_RULE_EN_GB']
GRAMMAR_ERROR_PENALTY = 0.75

# Cases:
# - Find best ASG summary without reference: similarity, penalties
# - Find best ASG summary with reference: similarity, reference BLEU, penalties
# - Score NN prediction: (similarity), reference BLEU
# - Compare NN prediction with ASG summary: reference BLEUs


class SummaryScorer:
    def __init__(self):
        self.pcn = ParseConceptNet(False)
        self.helper = Helper()
        self.language_checker = language_check.LanguageTool('en-GB')

    # TODO 3 scores: ASG vs NN, internal ASG, ASG vs target ASG
    def score(self, story, summary):
        similarity_score = self.similarity_score(story, summary)

        # TODO similarity/BLEU score compared to expected summary
        # TODO compare generated summaries with target ASG summary
        # TODO check if expected summary is in top 5/10 (hit) of generated summaries (use BLEU)
        # bleu_score = self.helper.bleu_score(story, summary)
        # bleu_score *= self.helper.count_sentences(story) / self.helper.count_sentences(summary)

        # TOOD use penalties to compare to NN (different score)
        grammar_penalty = self.grammar_penalty(story, summary)
        length_penalty = self.length_penalty(story, summary)

        overall_score = similarity_score * grammar_penalty * length_penalty
        # overall_score = similarity_score * bleu_score * grammar_penalty * length_penalty
        return round(overall_score, 1)

    def similarity_score(self, story, summary):
        tokenized_story = self.helper.tokenize_text(story, ignore_sentence=True)
        tokenized_summary = self.helper.tokenize_text(summary, ignore_sentence=True)

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
        # pp.pprint({k: dict(v) for k, v in dict(similar_words).items()})
        return summary_similarity

    # Decrease final score by 25% for every grammar error other than uncommon proper noun
    def grammar_penalty(self, story, summary):
        errors = self.language_checker.check(summary)
        num_errors = len(list(filter(lambda m: m.ruleId not in IGNORE_GRAMMAR_ERRORS, errors)))
        return GRAMMAR_ERROR_PENALTY ** num_errors

    # Divide final score by 2 for ever sentence over limit in summary
    def length_penalty(self, story, summary):
        story_length = self.helper.count_sentences(story)
        if story_length > 3:
            penalty = abs(self.helper.count_sentences(summary) - 3)
        else:
            penalty = abs(self.helper.count_sentences(summary) - story_length + 1)
        return 1 / (2 ** penalty)


def trim_batch_arg(lines):
    return list(map(lambda l: l[2:].replace(EOS_POSTFIX, ''), lines))


def process_args(args):
    if args.pair:
        pair = args.pair
        for i, pair_item in enumerate(pair):
            try:
                pair[i] = open(pair_item).read()
            except IOError:
                pass
        return tuple(zip(pair))
    batch_results = open(args.batch).read()
    lines = batch_results.split('\n')
    story_lines = list(filter(lambda l: l.startswith(BATCH_STORY_PREFIX), lines))
    summary_lines = list(filter(lambda l: l.startswith(BATCH_SUMMARY_PREFIX), lines))
    return tuple(map(trim_batch_arg, (story_lines, summary_lines)))


def parse_args():
    parser = argparse.ArgumentParser()
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('-b', '--batch', type=str, help='path to file with results of generating summaries')
    command_group.add_argument('-p', '--pair', type=str, nargs=2, help='strings or paths for story and summary')
    args = parser.parse_args()
    return process_args(args)


if __name__ == '__main__':
    summary_scorer = SummaryScorer()
    stories, summaries = parse_args()
    for story, summary in list(zip(stories, summaries)):
        score = summary_scorer.score(story, summary)
        print(story)
        print(summary)
        print(score)
        pass
