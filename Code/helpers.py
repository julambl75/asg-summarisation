import inflect


class Helpers:
    def __init__(self):
        self.p = inflect.engine()

    # Note: this does not work in general for nouns; for example p.plural_noun('cars') returns 'cars'
    def is_plural_pronoun(self, word):
        return self.p.plural_noun(word) == word

    def is_plural_verb(self, word):
        return self.p.plural_verb(word) == word

    # Add necessary predicates to ASG leaf nodes for edge cases
    def get_base_predicates(self, tag, lemma):
        if tag == 'prp':
            is_plural = self.is_plural_pronoun(lemma)
            return self._form_gram_num_predicate(lemma, is_plural)
        # elif tag == 'vbd':
        #     is_plural = self.is_plural_verb(lemma)
        #     return self._form_gram_num_predicate(lemma, is_plural)
        elif tag == 'vbg':
            return ' vb_obj_match(no_obj). '
        return ' '

    @staticmethod
    def _form_gram_num_predicate(word, is_plural):
        gram_num = 'plural' if is_plural else 'singular'
        return " {}({}). ".format(gram_num, word)
