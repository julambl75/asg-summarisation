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

PATH = os.path.dirname(os.path.abspath(__file__))

WORD_TYPES = ['adj', 'adv', 'noun', 'verb']


def read_data(word_type):
    wordbank = open(f'{PATH}/words/{word_type}.txt', encoding='utf-8').read().strip().split('\n')
    return list(map(lambda w: w.replace('_', ' '), wordbank))


def get_singular(word_type, word):
    assert word_type == 'noun'
    return word


def get_plural(word_type, word):
    assert word_type == 'noun'
    if word[-1:] == 'y':
        return word[:-1] + 'ies'
    return word


if __name__ == '__main__':
    words = {word_type: read_data(word_type) for word_type in WORD_TYPES}
    pass
