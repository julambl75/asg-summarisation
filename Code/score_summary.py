import collections
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
SCORE_COEFFICIENT = 100


class SummaryScorer:
    def __init__(self):
        self.pcn = ParseConceptNet(False)
        self.helper = Helper()
        self.language_checker = language_check.LanguageTool('en-GB')

    def asg_score(self, summaries, references=None):
        sorted_scores = []
        for summary in summaries:
            score = self.ttr(summary)
            score = int(score * SCORE_COEFFICIENT)
            sorted_scores.append((summary, score))
        sorted_scores.sort(key=itemgetter(1), reverse=True)

        # Check if reference summary is in top 5 (hit) of generated summaries
        if references:
            top_5 = list(map(itemgetter(0), sorted_scores[:5]))
            best_bleu = 0
            best = []
            for reference in references:
                for top_summary in top_5:
                    bleu_score = self.helper.bleu_score(reference, top_summary)
                    best_bleu = max(best_bleu, bleu_score)
                    best.append((bleu_score, reference, top_summary))
            print(sorted(best))
            assert best_bleu > SIMILAR_BLEU
        return sorted_scores

    # Computes the type-token ratio, a measure of word complexity
    @staticmethod
    def ttr(summary):
        words = [word.lower() for word in summary.split() if word not in string.punctuation]
        word_counts = collections.Counter(words)
        return len(word_counts) / len(words)
