import re
from collections import defaultdict

from pattern.en import conjugate, lemma, wordnet, singularize, pluralize
from pattern.en import NOUN
from wordfreq import word_frequency

LEXICAL_FIELD_REGEX = r'^\((\w+)\)'
SYNSET_DEPTH = 5


class QueryPattern:
    def __init__(self):
        # There is a bug with Python 3.7 causing the first call to Pattern to crash due to a StopIteration
        try:
            lemma('eight')
        except:
            pass

    @staticmethod
    def get_singular_noun(word):
        return singularize(word, pos=NOUN)

    @staticmethod
    def get_plural_noun(word):
        return pluralize(word, pos=NOUN)

    @staticmethod
    def lemmatize(word):
        return lemma(word)

    @staticmethod
    def conjugate(verb, person, tense, number):
        return conjugate(verb, person=person, tense=tense, number=number)

    def find_hypernym(self, word1, word2, return_plural=False):
        synsets1 = self._find_word_synsets(word1)
        synsets2 = self._find_word_synsets(word2)

        lexical_fields1 = self._find_lexical_fields(synsets1)
        lexical_fields2 = self._find_lexical_fields(synsets2)
        common_lexical_field = self._get_first_list_intersect(lexical_fields1, lexical_fields2)

        if common_lexical_field:
            return common_lexical_field

        hypernym = self._find_common_hypernym(synsets1, synsets2).replace('_', '-')
        if return_plural:
            hypernym_plural = pluralize(hypernym, pos=NOUN)
            if word_frequency(hypernym_plural.replace('-', ' '), 'en') > 0:
                return hypernym_plural
        return hypernym

    @staticmethod
    def _find_word_synsets(word):
        word = singularize(word, pos=NOUN)
        synsets = wordnet.synsets(word, pos=NOUN)
        return list(filter(lambda s: word.lower() == s.synonyms[0].lower(), synsets))

    @staticmethod
    def _find_lexical_fields(synsets):
        lexical_fields = []
        for synset in synsets:
            gloss = synset.gloss
            match = re.match(LEXICAL_FIELD_REGEX, gloss)
            if match:
                lexical_field = match.group(1)
                if lexical_field not in lexical_fields:
                    lexical_fields.append(lexical_field)
        return lexical_fields

    @staticmethod
    def _get_first_list_intersect(list1, list2):
        combined_indices = defaultdict(int)
        for list_items in [list1, list2]:
            for idx, item in enumerate(list_items):
                combined_indices[item] += idx
        if not combined_indices:
            return None
        return min(combined_indices.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _find_ordered_synsets_hypernyms(synsets):
        hypernym_hierarchies = defaultdict(int)
        for synset in synsets:
            hypernym_synsets = synset.hypernyms(recursive=True, depth=SYNSET_DEPTH)
            for i, hypernym_synset in enumerate(hypernym_synsets):
                hypernym = hypernym_synset.synonyms[0]
                hypernym_hierarchies[hypernym] = max(i, hypernym_hierarchies[hypernym])
        return hypernym_hierarchies

    def _find_common_hypernym(self, synset1, synset2):
        # Find hypernyms for each synset
        hypernyms1 = self._find_ordered_synsets_hypernyms(synset1)
        hypernyms2 = self._find_ordered_synsets_hypernyms(synset2)
        common_hypernyms = set(hypernyms1.keys()).intersection(set(hypernyms2.keys()))

        # Find most likely common hypernym
        hypernyms1 = {hypernym: idx for (hypernym, idx) in hypernyms1.items() if hypernym in common_hypernyms}
        hypernyms2 = {hypernym: idx for (hypernym, idx) in hypernyms2.items() if hypernym in common_hypernyms}
        common_hypernyms = {hypernym: (hypernyms1[hypernym] + hypernyms2[hypernym]) for hypernym in common_hypernyms}
        return min(common_hypernyms.items(), key=lambda x: x[1])[0]
