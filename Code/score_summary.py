import collections
import string
from operator import itemgetter

import language_check
import numpy as np

from helper import Helper
from parse_concept_net import ParseConceptNet

# Cases:
# - Find best ASG summary without reference: TTR
# - Find best ASG summary with reference: TTR, reference BLEU
# - Score NN prediction: TTR, reference BLEU, story BLEU?
# - Compare NN prediction with ASG summary: reference BLEUs

SIMILAR_BLEU = 0.70
SCORE_COEFFICIENT = 500
TOP_HIT_PERCENTILE = 75
PROPER_NOUN_SCORE_INC = 1


class SummaryScorer:
    def __init__(self):
        self.pcn = ParseConceptNet(False)
        self.helper = Helper()
        self.language_checker = language_check.LanguageTool('en-GB')

    def asg_score(self, story, summaries, references=None, proper_nouns=None, best_only=False):
        sorted_scores = []
        for summary in summaries:
            score = self.ttr_score(story, summary)
            score = int(score * SCORE_COEFFICIENT)
            sorted_scores.append((summary, score))

        if not sorted_scores:
            return [] if best_only else None

        # Increase score of summaries which start with proper noun
        if proper_nouns:
            for i, (summary, score) in enumerate(sorted_scores):
                if any(summary.startswith(proper_noun) for proper_noun in proper_nouns):
                    sorted_scores[i] = (summary, score + PROPER_NOUN_SCORE_INC)
        sorted_scores.sort(key=itemgetter(1), reverse=True)

        third_quartile_score = np.percentile(list(map(itemgetter(1), sorted_scores)), TOP_HIT_PERCENTILE)
        sorted_scores = [(summary, score) for summary, score in sorted_scores if score >= third_quartile_score]

        # Check if reference summary is in top quartile (according to score) of generated summaries
        if references:
            best_bleu = 0
            for reference in references:
                for top_summary, _ in sorted_scores:
                    bleu_score = self.helper.bleu_score(reference, top_summary)
                    best_bleu = max(best_bleu, bleu_score)
            assert best_bleu > SIMILAR_BLEU

        return sorted_scores[0] if best_only else sorted_scores

    # Computes a score based on type-token ratio, a measure of lexical density
    # Here the goal is to maximise the density of unique words (TTR), minimising summary length
    @staticmethod
    def ttr_score(story, summary):
        for punctuation in string.punctuation:
            story = story.replace(punctuation, '')
            summary = summary.replace(punctuation, '')
        story_words = [word.lower() for word in story.split()]
        summary_words = [word.lower() for word in summary.split()]
        word_counts = collections.Counter(summary_words)
        return len(word_counts) / (len(story_words) * len(summary_words))
