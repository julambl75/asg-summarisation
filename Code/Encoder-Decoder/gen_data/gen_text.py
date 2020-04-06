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

### Possesion:
# Joe/dog/rambunctious
# -> Joe had a rambunctious dog.
# Joe had a dog.
# The [hound] was rambunctious and [little].

### Doing something (work/sit/stand):
# TODO

### Getting dressed: Sarah put on her cape and make a phone call. / Peter put on his sweatshirt and went to the gym.
# TODO

###
# We need a large number of pairs for train (~9,000), a small number for test (~1,000), and a few for eval (~10).
###
import os

from pattern.en import conjugate, singularize, pluralize, referenced, lemma
from datamuse import datamuse

PATH = os.path.dirname(os.path.abspath(__file__))

WORD_TYPES = ['adj', 'adv', 'noun', 'verb', 'names']


def read_data(word_type):
    wordbank = open(f'{PATH}/words/{word_type}.txt', encoding='utf-8').read().strip().split('\n')
    return list(map(lambda w: w.replace('_', ' '), wordbank))


# >>> verb = "go"
# >>> conjugate(verb,
# ...     tense = "past",           # INFINITIVE, PRESENT, PAST, FUTURE
# ...    person = 3,                # 1, 2, 3 or None
# ...    number = "singular",       # SG, PL
# ...      mood = "indicative",     # INDICATIVE, IMPERATIVE, CONDITIONAL, SUBJUNCTIVE
# ...    aspect = "imperfective",   # IMPERFECTIVE, PERFECTIVE, PROGRESSIVE
# ...   negated = False)            # True or False
# u'went'

# When using pattern, there is a bug with Python 3.7 causing the first call to result in a StopIteration error
def initialize():
    global datamuse_api
    try:
        lemma('eight')
    except:
        pass
    datamuse_api = datamuse.Datamuse()


# https://www.clips.uantwerpen.be/pages/pattern-en#conjugation
def make_sentence(subject, verb, v_object, adjective=None, adverb=None):
    np = referenced(subject, article='indefinite')
    vp = conjugate(verb, tense='past', person=3, number='plural')
    return ' '.join((np, vp, adjective, pluralize(v_object), adverb))


# https://www.datamuse.com/api/
def make_summary(subject):
    np = referenced(subject, article='definite')
    vp = conjugate('be', tense='present', person=3, number='singular')
    # datamuse_api.words(rel_)


if __name__ == '__main__':
    initialize()

    words = {word_type: read_data(word_type) for word_type in WORD_TYPES}

    for i in range(10, 20):
        adjective = words['adj'][i]
        adverb = words['adv'][i]
        noun = words['noun'][i]
        noun2 = words['noun'][i + 20]
        verb = words['verb'][i]

        print(make_sentence(noun, verb, noun2, adjective, adverb))
