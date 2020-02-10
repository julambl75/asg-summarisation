from pycorenlp import StanfordCoreNLP
import nltk
from nltk.tree import *
import sys

SHOW_TREE = True
HIDE_TREE_ARG = '--no_tree'

PUNCTUATION = ['.', ',', ';', ':', '!', '?', "'", "''", '"', '``', '-']
POS_CATEGORIES = {'jj': 'adjective', 'nn': 'object', 'nns': 'object', 'nnp': 'object', 'nnps': 'object', \
                  'prp': 'subject', 'rb': 'adverb', 'rp': 'particle', 'vb': 'verb', 'vbd': 'verb', \
                  'vbg': 'verb', 'vbn': 'verb', 'vbp': 'verb', 'vbz': 'verb'}
# TODO differentiate object/subject and add missing tags
# TODO change how this is done

ROOT = 'ROOT'
SENTENCE = 'S'

asg_leaves = []  # ASG code for leaf nodes of tree
lemmas = {}  # Mapping of word -> lemma for ASG rules
constants = set()  # Pairs of (category, lemma) to create ILASP constants

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = StanfordCoreNLP('http://localhost:9000')

# Process whole documents
if len(sys.argv) > 1:
    try:
        text = open(sys.argv[1]).read()
    except IOError:
        text = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].strip() == HIDE_TREE_ARG:
        SHOW_TREE = False
else:
    text = "Please pass a filename or string as an argument."

output = nlp.annotate(text, properties={
    'annotators': 'tokenize,pos,lemma,depparse,parse',
    'outputFormat': 'json'
})

# Create mapping from words to lemmas
for sentence in output['sentences']:
    for token in sentence['tokens']:
        word = token['originalText'].lower()
        lemma = token['lemma'].lower()
        if word in lemmas.keys():
            assert lemmas[word] == lemma, "Multiple lemmas {} and {} for word {}".format(lemmas[word], lemma, word)
        else:
            lemmas[word] = lemma


# Remove all punctuation nodes
# TODO improve efficiency
def remove_punctuation(tree):
    i = 0
    tree_size = len(tree)
    while i < tree_size:
        node = tree[i]
        if not isinstance(node, nltk.Tree):
            return
        if node.label() in PUNCTUATION:
            del tree[i]
            tree_size -= 1
        else:
            remove_punctuation(node)
            i += 1


# Convert tree to ASG syntax
def tree_to_asg(tree, asg_leaves=asg_leaves):
    if isinstance(tree[0], nltk.Tree):  # non-leaf node
        if tree.label() != ROOT:
            child_labels = ' '.join([subtree.label() for subtree in tree if subtree.label() != SENTENCE])
        [tree_to_asg(subtree, asg_leaves) for subtree in tree]
    else:
        tag = tree.label().lower()
        word = tree[0].lower()
        predicates = ''
        if word in lemmas.keys() and tag in POS_CATEGORIES.keys():
            category = POS_CATEGORIES[tag]
            lemma = lemmas[word]
            constants.add((category, lemma))
            predicates = " {}({}). ".format(category, lemma)
        asg_leaves.append("{} -> \"{} \" {{{}}}".format(tag, word, predicates))
    return asg_leaves


# Extract parse trees
sentences = [Tree.fromstring(sentence) for sentence in [s['parse'] for s in output['sentences']]]
# tree = relabel_punc(sentences[0])
tree = sentences[0]

# Combine trees
itersentences = iter(sentences)
next(itersentences)
[tree.insert(len(tree[0]), sentence[0]) for sentence in itersentences]
remove_punctuation(tree)

# Print full tree (optional), leaf node ASG and ILASP constants
if SHOW_TREE:
    tree.pretty_print()
context_specific_asg = sorted(set(tree_to_asg(tree)))
[print(predicate) for predicate in context_specific_asg]
print('')
[print("#constant({},{}).".format(category,lemma)) for category, lemma in sorted(constants)]

sys.exit(0)  # Success
