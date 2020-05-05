import collections
import math
import string
from operator import itemgetter

import language_check

from helper import Helper
from parse_concept_net import ParseConceptNet

# Cases:
# - Find best ASG summary without reference: TTR
# - Find best ASG summary with reference: TTR, reference BLEU
# - Score NN prediction: TTR, reference BLEU, story BLEU?
# - Compare NN prediction with ASG summary: reference BLEUs

SIMILAR_BLEU = 0.75
SCORE_COEFFICIENT = 500


class SummaryScorer:
    def __init__(self):
        self.pcn = ParseConceptNet(False)
        self.helper = Helper()
        self.language_checker = language_check.LanguageTool('en-GB')

    def asg_score(self, story, summaries, references=None):
        sorted_scores = []
        for summary in summaries:
            score = self.ttr_score(story, summary)
            score = int(score * SCORE_COEFFICIENT)
            sorted_scores.append((summary, score))
        sorted_scores.sort(key=itemgetter(1), reverse=True)

        # Check if reference summary is in top 5 (hit) of generated summaries
        if references:
            top_5 = list(map(itemgetter(0), sorted_scores[:5]))
            best_bleu = 0
            for reference in references:
                for top_summary in top_5:
                    bleu_score = self.helper.bleu_score(reference, top_summary)
                    best_bleu = max(best_bleu, bleu_score)
            assert best_bleu > SIMILAR_BLEU
        return sorted_scores

    # Computes a score based on type-token ratio, a measure of lexical density
    @staticmethod
    def ttr_score(story, summary):
        for punctuation in string.punctuation:
            story = story.replace(punctuation, '')
            summary = summary.replace(punctuation, '')
        story_words = [word.lower() for word in story.split()]
        summary_words = [word.lower() for word in summary.split()]
        word_counts = collections.Counter(summary_words)
        return len(word_counts) / (len(story_words) * len(summary_words))
