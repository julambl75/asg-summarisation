# Introduction

In the approach that we have chosen, we use an ASG program, whose CFG represents the syntax of the English language, which ASP rules for the semantics.

# Parse Tree Generation

From the original text, we use CoreNLP to produce a parse tree whose leaf nodes are individual words, and each is assigned a POS (part-of-speech) tag. You can see an example of this if you run `python3 parse_core_nlp.py <TEXT>'.

# CFG Creation

From this we can then create a very basic inital CFG. The general structure is as follows:
- The root of our grammar is called `start` and contains a single `s_group` (sentence group).
- An `s_group` is recursively defined: it can either be empty, or contain a sentence `s` followed by another `s_group`.
- Each sentence is composed of a noun part `np` followed by a verb part `vp`.
- A `vp` in turn has a verb (different name depending on tense) as well as an `np`, and possibly a verb in continuous form `vbg` and/or and adverb `rb`.
- Each `np` may have a determinant `dt` (if singular), can have an adjective `jj`, an adverb, and always a noun or pronoun.
- Leaf nodes are thus nouns/pronouns, verbs, adjectives, adverbs and determinants.

# Semantic Rules

For each individual word (leaf node), we we have the word as a string in our CFG (with a space at the end to make it more readable), and enocode its semantics as ASP rules (as described below).

At the moment, all the semantic information is added in manually, but we will use the ConceptNET API to do so eventually.

## Background

In order to pass all of this information up the tree, we define in `#background` a binary predicate `property` for all of the unary predicates that describe properties of nodes (see below). This binary predicate takes the name of the unary predicate as its first argument and the value as its second argument. For example, `color(red)` will become `property(color,red)` for any node it is defined in. This way, we simply need to pass every `property` up the tree, rather than any possible unary predicate describing a property.

In the particular case of some adjectives (such as colors), we assign the predicate `property(n_adj_match,P)` (where `P` would be `color`), along with a unary predicate `can_have(P)`. This allows us the restrict the use of such adjectives to only some nouns.

## Leaf Node Level

Every noun/pronoun can be an `object` and/or a `subject`. In addition, specific actions that can be performed on objects using a verb are specified with the unary predicate `vb_obj_match`. For example, we may want to encode that a car can be driven using `vb_obj_match(driveable)`, or that the verb "to rain" should not be paired with an object.

For verbs, we have different labels depending on the form it is in, but they always will have the unary predicate `verb`, whose argument is the verb in its base form. Also, some verbs will need to have `vb_obj_match`, as we do not want to drive a desk for example.

Adjectives are used to describe some property of objects, so they have a unary predicate which corresponds to this property. For example, the adjective "green" needs to have the predicate `color(green)`.

## Noun Group Level

At the noun level, we specify for all nouns (and not pronouns) that verbs must be conjugated in the third person, using `gram_pers(third)`.

For nouns with adjectives, this is where we check that adjectives requiring `n_adj_match` are paired with a noun that can_have` this category of adjective.

## Verb Group Level

For different conjugations of verbs, we use a different label in the CFG. From this, we are able to pass the unary predicate `tense` (with value `present` or `past`), which will help us later on.

For verbs which can only be applied to some nouns, this is where we ensure that any verb which has the property `vb_obj_match` is paired with a noun group which also has the property.

Finally, we do not allow any verb groups which do not have a valid `object` or `adverb`.

## Sentence Level

A sentence must always have a subject, so we check that the noun part is indeed one.

To ensure subject-verb agreement in the present tense, we use constraints to check that the former has `gram_pers(third)` iff the later does too. For some verbs (for instance "to be"), the conjugation is different for singular and plural subjects, so we have unary predicates `singular` and `plural` defined at the verb level, as well as at the noun group or pronoun level.

In order to keep track of what information the original text contains, we can define the 4-ary predicate `action`, which respectively takes a subject, verb and object. There also exists a more specific 5-ary predicate with the same name, which simply takes an adjective as an additional parameter. This is then used in a constraint, which restricts the subparts of a given sentence to match the information passed via `action`. For example, with the single predicate `action(mary,eat,apple,past)`, a sentence is restricted to say that Mary ate one or more apples.

At the moment the grammar supports five different sentence structures:
- subject-verb-object
- subject-verb-object-object
- subject-verb-adverb
- subject-verb-adjective
- subject-verb-adjective-object

## Sentence Group Level

At the sentence group level, we do not want to talk about the same subject, verb and/or action in two consecutive sentences in the summary, so we rule these cases out via a constraint.

In the same way, we avoid having a sentence in the past tense after one that is in the present.

Finally, we use a unary predicate `count` to keep track of the number of sentences in our summary. For a non-empty sentence group we recursively increment the value, otherwise it is initialized to 0.

## Root Node

At the root node, we check that our sentence group has a `count` which is at most 2, as we need to keep the summart succinct.

# Learning

## Actions

In order to learn facts from sentences, we use positive examples formatted as lists of words. From these we are able to learn sentence level atoms using the predicate `action`, which we specify as `#modeh` (in ILASP syntax). To support this learning, we define every `#constant` as well. For instance, we use `#constant(verb,like)` so that we can use `const(verb)` as an argument in the `#modeh`.

## Properties

In addition, we are able to learn properties of nouns with respect to adjectives. Do do this, we use `#modeh(can_have,P)`, where `P` is for instance `color`. This way, we don't have to specify as much information in the leaf node level if this is not required. Also, it may be useful to capture edge cases where a type of adjective is applied on a particular noun, but we have not captured that this is possible.

## Narrative

In order to teach our program how to understand a narrative, we have started out with a unary predicate `direction`, which can take values `forward`, `backward` and `noop`. This can be assigned to verbs, so that if we encounted a sentence with `direction(forward)` and then one with `direction(backward)`, we can replace them with a sentence that has `direction(noop)`.

# Summary Generation

In order to generate a summary, we have a script called `main.py`, which takes a text as input. From this, it splits it into sentences, which get divided into words. For each sentence, a list of its constituent words is appended as a positive example to our ASG program. We do not create a single positive example for the entire text, otherwise our program would not be able to learn from it, due to the restriction on the number of sentences in a summary (thus we can use the same script without changing any rules). In order to be able to learn about the actions that take place in the story, we use the parse tree to create ILASP constants for each word; these then also get appended to the ASG program.

At this stage we run `asg` in learning mode, to generate a new program that contains the `action` predicates learned from the text.

When building up new summary generation rules, we slightly change the mode bias, allowing us to learn rules to give summaries based on `action` predicates and additional words from the text. This gives us rules for the predicate `summary`.  

Then, we run this new program in run mode, which finally lists the possible summaries.

# Preprocessor

## Idea

The goal of the `Preprocessor` is to simplify the original story to make the job easier and faster for ASG. The first thing it does is create a text relationship map for all related words (using ConceptNET). From this, it is able to create one for sentences by summing the weights of words on a per-sentence basis.

Now that we can quantify how "related" sentences are, we sum all of these outgoing links for every sentence, dividing this by a normalizing constant to penalize longer sentences (as they have a better chance to contain words that are similar to words in other sentences). The more "linked" a sentence is, the more relevant its meaning to the story, so the more important it is for the summary.

**(Simplification 1)** What the `Preprocessor` then does is to drop sentences which have a normalized importance that is lower than the 25th percentile, effectively keeping only the more "relevant" ones.

**(Simplification 2)** Its final step is to replace every word in a synonym pair (weight >= 2 according to ConceptNET) with its shorter counterpart, if possible.

## Search Space Reduction

Running time in ILASP is proportional to the search space size, so by reducing the latter we can make ASG faster.

By dropping sentences which are irrelevant for the summary and homogenizing the text, we are able to greatly reduce the search space, as can be seen in the below examples. In both cases, we will refer to the search space for learning the `action` predicates (see above).

### Example 1: Peter Little

See the original story in Samples/peter_little/peter_little_full.txt.

|  | Original | Simplification 1 | Simplification 2 |
|-----------------------------|:--------:|:----------------:|:----------------:|
| ASG tree leaf nodes |  |  |  |
| Constants for learning task |  |  |  |
| Search space size |  |  |  |

### Example 2: Elon

See the original story in Samples/elon/elon.txt.

|  | Original | Simplification 1 | Simplification 2 |
|-----------------------------|:--------:|:----------------:|:----------------:|
| ASG tree leaf nodes | 14 | ? | 8 |
| Constants for learning task | 17 | ? | 9 |
| Search space size | 2016 | ? | 120 |
