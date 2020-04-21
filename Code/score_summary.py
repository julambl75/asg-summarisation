from collections import defaultdict
from operator import itemgetter

import language_check

from helper import Helper
from parse_concept_net import ParseConceptNet

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

    def asg_score(self, story, summaries, reference=None):
        sorted_scores = []
        for summary in summaries:
            score = self._asg_score(story, summary, reference)
            sorted_scores.append((summary, score))
        sorted_scores.sort(key=itemgetter(1), reverse=True)

        # Check if reference summary is in top 5 (hit) of generated summaries
        if reference:
            top_5 = list(map(itemgetter(0), sorted_scores[:5]))
            assert reference in top_5
        return sorted_scores

    def _asg_score(self, story, summary, reference=None):
        score = self._similarity_score(story, summary)
        score *= self._grammar_penalty(summary)
        score *= self._length_penalty(story, summary)

        if reference:
            score *= self.helper.bleu_score(summary, reference)
        return round(score, 1)

    def _similarity_score(self, story, summary):
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
    def _grammar_penalty(self, summary):
        errors = self.language_checker.check(summary)
        num_errors = len(list(filter(lambda m: m.ruleId not in IGNORE_GRAMMAR_ERRORS, errors)))
        return GRAMMAR_ERROR_PENALTY ** num_errors

    # Divide final score by 2 for ever sentence over limit in summary
    def _length_penalty(self, story, summary):
        story_length = self.helper.count_sentences(story)
        if story_length > 3:
            penalty = abs(self.helper.count_sentences(summary) - 3)
        else:
            penalty = abs(self.helper.count_sentences(summary) - story_length + 1)
        return 1 / (2 ** penalty)
