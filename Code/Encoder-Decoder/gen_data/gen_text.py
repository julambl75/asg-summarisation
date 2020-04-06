# 'The flowers are blooming now' # DT NNS V VING ADV
# 'The leaves are growing'
# 'Spring is coming very soon'
# -> 'Spring is coming'
#
# 'The wall is covered in poison ivy' # DT NN V VING IN NN
# 'At night the house is full of ghosts'
# 'You can hear voices coming from inside'
# 'The house must be haunted'
# -> 'At night the house is full of ghosts so it is haunted'
#
# 'Suzanne went to cooking school'
# 'Sophie learned to cook in Australia'
# 'They are owners of a famous restaurant in Melbourne'
# -> 'Sophie and Suzanne are renowned chefs' # NNP CC NNP V ADJ NNS

###
# We need a large number of pairs for train (~9,000), a small number for test (~1,000), and a few for eval (~10).
###
import os

from pattern.en import conjugate, singularize, pluralize, referenced

PATH = os.path.dirname(os.path.abspath(__file__))

WORD_TYPES = ['adj', 'adv', 'noun', 'verb']


def read_data(word_type):
    wordbank = open(f'{PATH}/words/{word_type}.txt', encoding='utf-8').read().strip().split('\n')
    return list(map(lambda w: w.replace('_', ' '), wordbank))


# https://www.clips.uantwerpen.be/pages/pattern-en#conjugation

# >>> verb = "go"
# >>> conjugate(verb,
# ...     tense = "past",           # INFINITIVE, PRESENT, PAST, FUTURE
# ...    person = 3,                # 1, 2, 3 or None
# ...    number = "singular",       # SG, PL
# ...      mood = "indicative",     # INDICATIVE, IMPERATIVE, CONDITIONAL, SUBJUNCTIVE
# ...    aspect = "imperfective",   # IMPERFECTIVE, PERFECTIVE, PROGRESSIVE
# ...   negated = False)            # True or False
# u'went'

def make_sentence(subject, verb, v_object, adjective=None, adverb=None):
    np = referenced(subject, article='indefinite')
    vp = conjugate(verb, tense='past', person=3, number='plural') + ' ' + adjective + ' ' + pluralize(v_object)
    return np + ' ' + vp + ' ' + adverb


if __name__ == '__main__':
    words = {word_type: read_data(word_type) for word_type in WORD_TYPES}

    adjective = words['adj'][15]
    adverb = words['adv'][15]
    noun = words['noun'][15]
    noun2 = words['noun'][35]
    verb = words['verb'][15]

    print(make_sentence(noun, verb, noun2, adjective, adverb))
