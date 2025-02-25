import collections
import string
from operator import itemgetter

import numpy as np

from helper import Helper
from parse_concept_net import ParseConceptNet


TTR_IGNORE = {'a', 'the', 'be', 'being', 'is', 'am', 'are', 'was', 'were'}
TTR_IGNORE_MIN_FREQU_RATIO = 0.4

SIMILAR_BLEU = 0.65
SCORE_COEFFICIENT = 500
TOP_HIT_PERCENTILE = 75
PROPER_NOUN_SCORE_INC = 2


class SummaryScorer:
    def __init__(self):
        self.pcn = ParseConceptNet(False)
        self.helper = Helper()

    def asg_score(self, story, summaries, references=None, proper_nouns=None):
        # Find common words in story which are find if repeated in summary
        story_len = sum([1 for sent in story.split('.') if len(sent)])
        story_words, word_counts = self.get_words_and_counts(story)
        most_common_words = word_counts.most_common()
        highest_word_count = most_common_words[0][1]

        story_topics = set()
        if highest_word_count > 1 and highest_word_count >= story_len * TTR_IGNORE_MIN_FREQU_RATIO:
            story_topics = {word for word, count in most_common_words if count == highest_word_count}
            for proper_noun in proper_nouns:
                for word in proper_noun.lower().split():
                    story_topics.add(word)

        sorted_scores = []
        for summary in summaries:
            score = self.ttr_score(story_words, summary, story_topics)
            score = int(score * SCORE_COEFFICIENT)
            sorted_scores.append((summary, score))

        if not sorted_scores:
            return None

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

        return sorted_scores

    @staticmethod
    def get_words_and_counts(text, ignore_words=TTR_IGNORE):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        words = [word.lower() for word in text.split() if word.lower() not in ignore_words]
        counts = collections.Counter(words)
        return words, counts

    # Computes a score based on type-token ratio, a measure of lexical density
    # Here the goal is to maximise the density of unique words (TTR), minimising summary length
    # If some words appear frequently in the story (topics), we do not want this to hinder prioritising good summaries
    def ttr_score(self, story_words, summary, story_topics):
        ignore_counts = story_topics.union(TTR_IGNORE)
        summary_words, word_counts = self.get_words_and_counts(summary, ignore_counts)
        score_scale = len(story_words) * len(summary_words) or 1
        return len(word_counts) / score_scale
