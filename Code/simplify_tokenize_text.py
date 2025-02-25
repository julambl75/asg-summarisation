import re
from itertools import chain
from operator import itemgetter

import contractions

from helper import Helper
from parse_concept_net import ParseConceptNet
from parse_core_nlp import SUBORDINATING_CONJUNCTIONS
from query_pattern import QueryPattern

SUBORDINATING_CONJUNCTIONS_JOINED = '|'.join(SUBORDINATING_CONJUNCTIONS)
SUBORDINATING_CONJUNCTIONS_REGEX = f'(?i) ?({SUBORDINATING_CONJUNCTIONS_JOINED}) '

SUBSTITUTIONS = {'an': 'a'}
EOS_REPLACE = ['!', ',', ';', ':']
EOS_REMOVE = ['?']
EOS_REMOVE_INNER = ['–', '—']
EOS = '.'
HYPHEN = '-'

ADVERB_POS = 'RB'
CONJUNCTIVE_POS = 'CC'
VERB_POS = 'VB'
PREPOSITION_POS = 'IN'
PRONOUN_POS = 'PRP'
COMMON_NOUN_POS = 'NN'
COMMON_NOUN_POS_PL = 'NNS'
PROPER_NOUN_POS = 'NNP'
PROPER_NOUN_POS_PL = 'NNPS'

PAST_TENSE_VERB_POS = 'VBD'
COMPLEX_CLAUSE_AUX_VERB_POS = 'VBN'
COMPLEX_CLAUSE_SUBSTITUTIONS = {'a': 'the'}
EOS_TOKENIZED = ('.', '.')

SUPERFLUOUS_POS = ['PRP$', 'UH']  # Possessive pronouns and interjections
SUPERFLUOUS_WORDS = ['some']
DEPENDENT_CLAUSE_POS = 'WRB'  # When, where, which...
SUBJECT_POS = ['NN', 'NNS', 'NNP', 'NNPS', 'EX', 'PRP']

PERSON_PRONOUNS_SG = ['he', 'she']
PERSON_PRONOUNS_PL = ['they']


class TextSimplifier:
    def __init__(self, text):
        self.text = text
        self.proper_nouns = set()

        self.helper = Helper()
        self.query_pattern = QueryPattern()
        self.pcn = ParseConceptNet(False)

    def tokenize(self):
        self._add_final_punctuation()
        self._substitute_determiners()
        self._expand_contractions()
        self._replace_punctuation()
        self._remove_subordinating_conjunctions()

        tokenized = self.helper.tokenize_text(self.text)
        tokenized = self._change_single_vbn_to_vbd(tokenized)
        tokenized = self._remove_punctuated_acronyms(tokenized)
        tokenized = self._combine_complex_nouns(tokenized)
        tokenized = self._move_adverbs_to_end(tokenized)
        tokenized = self._split_conjunctive_clauses(tokenized)
        tokenized = self._expand_complex_clauses(tokenized)
        tokenized = self._replace_noun_conjunction_with_superclass(tokenized)
        tokenized = self._remove_superfluous_words(tokenized)
        tokenized = self._separate_dependant_clauses(tokenized)
        tokenized = self._remove_preposition_clauses(tokenized)
        tokenized = self._remove_verbless_sentences(tokenized)
        tokenized = self._substitute_pronouns_for_proper_nouns(tokenized)

        self._replace_story_new_tokenized(tokenized)
        return tokenized, self.text, self.proper_nouns

    def _add_final_punctuation(self):
        self.text = self.text.strip()
        if self.text[-1] != EOS:
            self.text += EOS

    def _substitute_determiners(self):
        for key, value in SUBSTITUTIONS.items():
            self.text = re.sub(r"\b{}\b".format(key), value, self.text)

    # Ex: It's a lot of fun. We're here. -> It is a lot of fun. We are here.
    def _expand_contractions(self):
        self.text = contractions.fix(self.text, leftovers=False, slang=False)

    # Ex: Why are you doing this? He liked her; she liked him.
    #  -> He liked her. she liked him.
    # Ex: This is a tree, it is a big tree. The car – the long one – was green. It is convenient, is it not nice? Yay!
    #  -> This is a tree. it is a big tree. The car was green. it is convenient. Yay.
    def _replace_punctuation(self):
        # Replace exclamation mark, comma, semi-colon and colon with full stop
        for punctuation in EOS_REPLACE:
            self.text = self.text.replace(punctuation, EOS)

        # Remove independent clauses in between dashes
        for punctuation in EOS_REMOVE_INNER:
            while punctuation in self.text:
                clause_start = self.text.index(punctuation)
                clause_end = self.text[clause_start:].index(EOS) + clause_start
                # Check if clause ends with another dash or with full stop
                if punctuation in self.text[clause_start + 1:clause_end]:
                    clause_end = self.text[clause_start + 1:clause_end].index(punctuation) + clause_start + 1
                self.text = self.text[:clause_start] + self.text[clause_end + 2:]

        # Remove sentence clauses
        for punctuation in EOS_REMOVE:
            while punctuation in self.text:
                punctuation_idx = self.text.index(punctuation)
                text_until_punctuation = self.text[:punctuation_idx]
                sentence_start = 0
                if EOS in text_until_punctuation:
                    sentence_start = len(text_until_punctuation) - text_until_punctuation[::-1].index(EOS)
                self.text = self.text[:sentence_start] + self.text[punctuation_idx + 1:]
        if self.text[0] == ' ':
            self.text = self.text[1:]

    # Ex: She never walks alone after sunset because she is afraid of the dark.
    #  -> She never walks alone. sunset. she is afraid of the dark.
    def _remove_subordinating_conjunctions(self):
        self.text = re.sub(SUBORDINATING_CONJUNCTIONS_REGEX, EOS, self.text)

    # Ex: The system processed (VBN) a ticket. -> The system processed (VBD) a ticket.
    def _change_single_vbn_to_vbd(self, tokenized):
        for sentence in tokenized:
            sentence_verbs = self._tokens_to_pos(sentence, VERB_POS)
            if len(sentence_verbs) == 1:
                verb_lemma, verb_pos = sentence_verbs[0]
                if verb_pos == COMPLEX_CLAUSE_AUX_VERB_POS:
                    verb_idx = sentence.index((verb_lemma, verb_pos))
                    sentence[verb_idx] = (verb_lemma, PAST_TENSE_VERB_POS)
        return tokenized

    # Ex: Mrs.   -> Mrs
    # Ex: U.S.A. -> USA
    @staticmethod
    def _remove_punctuated_acronyms(tokenized):
        for sentence in tokenized:
            for i, (word, pos) in enumerate(sentence):
                if (word, pos) != EOS_TOKENIZED:
                    word = word.replace(EOS, '')
                    sentence[i] = (word, pos)
        return tokenized

    # Ex: Peter Little -> PeterLittle
    # Ex: They built a birdhouse. A bird is in the bird house. -> They built a birdhouse. A bird is in the birdhouse.
    # Ex: A bird is in the bird house. -> A bird is in the bird-house.
    def _combine_complex_nouns(self, tokenized):
        text_nouns = {word for word, _ in self._tokens_to_pos(chain.from_iterable(tokenized), COMMON_NOUN_POS)}
        for sentence in tokenized:
            i = 0
            while i < len(sentence) - 1:
                (word, pos) = sentence[i]
                (next_word, next_pos) = sentence[i + 1]
                # Reduce ASG search space
                if pos.startswith(COMMON_NOUN_POS) and next_pos.startswith(COMMON_NOUN_POS):
                    new_word = word + next_word
                    if pos == COMMON_NOUN_POS:
                        # Add hyphen unless combined words normally don't have one
                        if new_word not in text_nouns:
                            new_word = word + HYPHEN + next_word
                    sentence[i] = (new_word, pos)
                    sentence.pop(i + 1)
                else:
                    i += 1
        return tokenized

    # Ex: Sometimes it is easy.                   -> it is easy Sometimes.
    # Ex: He always studied and did his homework. -> He studied always and did his homework.
    # Ex: He studied and always did his homework. -> He studied and did his homework always.
    def _move_adverbs_to_end(self, tokenized):
        for i, sentence in enumerate(tokenized):
            pos_tags = self._get_pos_tags(sentence)

            if ADVERB_POS in pos_tags:
                adverb_idx = pos_tags.index(ADVERB_POS)
                # Adverb is already at the end
                if adverb_idx == len(sentence) - 1:
                    continue
                adverb_new_idx = sentence.index(EOS_TOKENIZED) - 1
                if CONJUNCTIVE_POS in pos_tags:
                    conjunction_idx = pos_tags.index(CONJUNCTIVE_POS)
                    # Adverb needs to be put right before conjunction
                    if adverb_idx < conjunction_idx:
                        adverb_new_idx = conjunction_idx - 1
                adverb_token = sentence.pop(adverb_idx)
                sentence.insert(adverb_new_idx, adverb_token)
        return tokenized

    # Ex: We looked left and they saw us. -> We looked left. they saw us.
    # Ex: Cars have wheels and go fast.   -> Cars have wheels. Cars go fast.
    def _split_conjunctive_clauses(self, tokenized):
        i = 0
        while i < len(tokenized):
            sentence = tokenized[i]
            pos_tags = self._get_pos_tags(sentence)

            if CONJUNCTIVE_POS in pos_tags:
                conjunct_idx = pos_tags.index(CONJUNCTIVE_POS)
                first_clause = sentence[:conjunct_idx]
                second_clause = sentence[conjunct_idx + 1:]

                first_clause_verbs = self._tokens_to_pos(first_clause, VERB_POS)

                # Subject has been omitted from second clause
                if first_clause_verbs and second_clause[0][1].startswith(VERB_POS):
                    first_clause_verb_idx = first_clause.index(first_clause_verbs[0])
                    first_clause_subject = first_clause[:first_clause_verb_idx]
                    second_clause = first_clause_subject + second_clause
                # Conjuncts do not both have their own verb, keep conjunction the way it is
                elif not (self._tokens_to_pos(first_clause, VERB_POS) and self._tokens_to_pos(second_clause, VERB_POS)):
                    i += 1
                    continue
                tokenized[i] = first_clause + [EOS_TOKENIZED]
                tokenized.insert(i + 1, second_clause)
            i += 1
        return tokenized

    # Ex: There was a boy named Peter. -> There was a boy. the boy was named Peter.
    def _expand_complex_clauses(self, tokenized):
        i = 0
        while i < len(tokenized):
            sentence = tokenized[i]
            pos_tags = self._get_pos_tags(sentence)
            aux_clause_idx = self.helper.find_default(pos_tags, COMPLEX_CLAUSE_AUX_VERB_POS)
            main_clause_verbs = any(self._tokens_to_pos(sentence[:aux_clause_idx], VERB_POS))

            # If there is after a verb clause an auxiliary clause (starting with a VBN that does not follow a verb)
            if main_clause_verbs and aux_clause_idx > 1 and not pos_tags[aux_clause_idx - 1].startswith(VERB_POS):
                main_clause = sentence[:aux_clause_idx]
                aux_clause_obj = sentence[aux_clause_idx:]

                # Prepend to the auxiliary clause everything in the main clause after its last verb
                #   i.e., use the main clause's object as the subject of the auxiliary clause
                pos_tags_main = self._get_pos_tags(main_clause)
                main_clause_verbs = self._tokens_to_pos(main_clause, VERB_POS)
                main_clause_obj_idx = len(main_clause) - pos_tags_main[::-1].index(main_clause_verbs[-1][1])
                main_clause_obj = main_clause[main_clause_obj_idx:]

                # Change 'a' to 'the' at start of subject of the auxiliary sentence
                first_aux_word, first_aux_pos = main_clause_obj[0]
                if first_aux_word in COMPLEX_CLAUSE_SUBSTITUTIONS.keys():
                    first_aux_word = COMPLEX_CLAUSE_SUBSTITUTIONS[first_aux_word]
                    main_clause_obj[0] = (first_aux_word, first_aux_pos)
                aux_clause = main_clause_obj + main_clause_verbs + aux_clause_obj

                # Replace tokenized sentence with two tokenized sentences and check auxiliary clause next
                tokenized[i] = main_clause + [EOS_TOKENIZED]
                tokenized.insert(i + 1, aux_clause)
            i += 1
        return tokenized

    # Ex: She raised chickens and geese. -> She raised poultry.
    # Ex: He was interested in stars and planets. -> He was interested in astronomy.
    # Ex: They like cars and trucks. -> They like motor-vehicles.
    def _replace_noun_conjunction_with_superclass(self, tokenized):
        for sentence in tokenized:
            pos_tags = self._get_pos_tags(sentence)
            first_noun_sg_idx = self.helper.find_default(pos_tags, COMMON_NOUN_POS)
            first_noun_pl_idx = self.helper.find_default(pos_tags, COMMON_NOUN_POS_PL)
            first_noun_idx = max(first_noun_sg_idx, first_noun_pl_idx)
            # Check if next token is conjunction and following one is another common noun
            if 0 < first_noun_idx < len(pos_tags) - 3:
                (word, pos) = sentence[first_noun_idx]
                next_pos = pos_tags[first_noun_idx+1]
                (next_next_word, next_next_pos) = sentence[first_noun_idx+2]
                if next_pos == CONJUNCTIVE_POS and next_next_pos in [COMMON_NOUN_POS, COMMON_NOUN_POS_PL]:
                    plural_comb = pos == COMMON_NOUN_POS_PL or next_next_pos == COMMON_NOUN_POS_PL
                    hypernym = self.query_pattern.find_hypernym(word, next_next_word, return_plural=True)
                    if hypernym:
                        similarity = all(self.pcn.compare_words(hypernym.replace('-', ' '), w, True) for w in (word, next_next_word))
                        if similarity:
                            comb_pos = COMMON_NOUN_POS_PL if plural_comb else COMMON_NOUN_POS
                            sentence.pop(first_noun_idx)
                            sentence.pop(first_noun_idx)
                            sentence[first_noun_idx] = (hypernym, comb_pos)
        return tokenized

    # Ex: She ate her chocolate. -> She ate chocolate.
    # Ex: She ate some chocolate. -> She ate chocolate.
    def _remove_superfluous_words(self, tokenized):
        for sentence in tokenized:
            words = self._get_words_downcase(sentence)
            pos_tags = self._get_pos_tags(sentence)

            for superfluous_tag in SUPERFLUOUS_POS:
                while superfluous_tag in pos_tags:
                    superfluous_idx = pos_tags.index(superfluous_tag)
                    sentence.pop(superfluous_idx)
                    pos_tags = self._get_pos_tags(sentence)

            for superfluous_word in SUPERFLUOUS_WORDS:
                while superfluous_word in words:
                    superfluous_idx = words.index(superfluous_word)
                    sentence.pop(superfluous_idx)
                    words = self._get_words_downcase(sentence)

            if pos_tags[0] == PREPOSITION_POS:
                sentence.pop(0)
        return tokenized

    # Ex: I want to be President when I grow up. -> I want to be President.
    # Ex: When I grow up, I will have a garden. -> I will have a garden.
    # Ex: When I grow tomatoes it will be a good moment. -> DELETED
    def _separate_dependant_clauses(self, tokenized):
        i = 0
        while i < len(tokenized):
            sentence = tokenized[i]
            pos_tags = self._get_pos_tags(sentence)
            if DEPENDENT_CLAUSE_POS in pos_tags:
                # When the wh-adverb is at the start of a sentence, there is usually a comma after the dependant clause,
                # Because we replace commas with full stops, the dependant clause will already be its own sentence.
                if pos_tags[0] == DEPENDENT_CLAUSE_POS:
                    tokenized.pop(i)
                    continue
                else:
                    dependant_idx = pos_tags.index(DEPENDENT_CLAUSE_POS)
                    main_clause = sentence[:dependant_idx]
                    tokenized[i] = main_clause + [EOS_TOKENIZED]
            i += 1
        return tokenized

    # Ex: They hang it up in a tree. -> They hang it up.
    def _remove_preposition_clauses(self, tokenized):
        for i, sentence in enumerate(tokenized):
            pos_tags = self._get_pos_tags(sentence)
            if PREPOSITION_POS in pos_tags:
                preposition_idx = pos_tags.index(PREPOSITION_POS)
                verb_indices = [idx for idx, tag in enumerate(pos_tags) if tag.startswith(VERB_POS)]
                # Preposition must be after the verb's object
                if verb_indices and preposition_idx > verb_indices[-1] + 2:
                    tokenized[i] = sentence[:preposition_idx] + [EOS_TOKENIZED]
        return tokenized

    # Ex: Spectacular discovery. -> DELETED
    def _remove_verbless_sentences(self, tokenized):
        i = 0
        while i < len(tokenized):
            sentence = tokenized[i]
            verbs = self._tokens_to_pos(sentence, VERB_POS)
            if not verbs:
                tokenized.pop(i)
            else:
                i += 1
        return tokenized

    # Ex: Mary is drinking coffee. She is angry.
    #  -> Mary is drinking coffee. Mary is angry.
    # Ex: Antonio is a cheesemaker. He makes burrata. Italians eat pasta. They make it with egg sometimes.
    #  -> Antonio is a cheesemaker. Antonio makes burrata. Italians eat pasta. Italians make it with egg sometimes.
    def _substitute_pronouns_for_proper_nouns(self, tokenized):
        proper_nouns_sg = set()
        proper_nouns_pl = set()
        for sentence in tokenized:
            for word, pos in sentence:
                if pos == PROPER_NOUN_POS:
                    proper_nouns_sg.add(word)
                if pos == PROPER_NOUN_POS_PL:
                    proper_nouns_pl.add(word)
        self.proper_nouns.update(proper_nouns_sg)
        self.proper_nouns.update(proper_nouns_pl)

        pronouns = self._tokens_to_pos(chain.from_iterable(tokenized), PRONOUN_POS)
        pronouns_pl = {pronoun.lower() for pronoun, _ in pronouns if pronoun.lower() in PERSON_PRONOUNS_PL}
        # Support singular pronoun 'they' when gender of person is unknown
        if len(pronouns_pl) == 0:
            PERSON_PRONOUNS_PL.extend(PERSON_PRONOUNS_SG)
        pronouns_sg = {pronoun.lower() for pronoun, _ in pronouns if pronoun.lower() in PERSON_PRONOUNS_SG}

        pronouns_to_proper = {}
        if len(proper_nouns_sg) == 1 and len(pronouns_sg) == 1:
            pronouns_to_proper[self._get_first_elem(pronouns_sg)] = self._get_first_elem(proper_nouns_sg)
        if len(proper_nouns_pl) == 1 and len(pronouns_pl) == 1:
            pronouns_to_proper[self._get_first_elem(pronouns_pl)] = self._get_first_elem(proper_nouns_pl)
        for sentence in tokenized:
            for i, (word, pos) in enumerate(sentence):
                word = word.lower()
                if pos == PRONOUN_POS and word in pronouns_to_proper.keys():
                    sentence[i] = (pronouns_to_proper[word], pos)
        return tokenized

    def _replace_story_new_tokenized(self, tokenized):
        tokenized_words = list(map(itemgetter(0), chain.from_iterable(tokenized)))
        self.text = ' '.join(tokenized_words).replace('. .', EOS).replace(' .', EOS)

    @staticmethod
    def _tokens_to_pos(tokens, pos):
        return list(filter(lambda t: t[1].startswith(pos), tokens))

    @staticmethod
    def _get_words_downcase(sentence):
        return list(map(lambda w: w.lower(), map(itemgetter(0), sentence)))

    @staticmethod
    def _get_pos_tags(sentence):
        return list(map(itemgetter(1), sentence))

    @staticmethod
    def _get_first_elem(values):
        assert len(values) > 0
        return next(iter(values))
