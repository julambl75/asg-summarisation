## To show Alessandra

- Improved language support
- Constraints for multi-sentence summaries
- TTR for scoring
- Summary generation for NN training examples

## TODO

- Fix Preprocessor so simplify complex sentences and filter less out
- Support peter_little.txt
- Support puzzle.txt
- Train NN on random stories

## For report

- Describe action predicates as high level semantic descriptor of all possible actions that can happen in sentences
- Think about initial motivation (have definition of what is summary, NN does not)
    1. NNs need lots of data and time to summarize
    2. ASG can give results with short list of rules, pre/post-processing and carefully constructed structure
- Maybe formalize mathematically task of summarization (with CFG, BK, E+, E-)
    1. CFG is language, BK is leaf nodes, result is actions
    2. CFG is language, BK is leaf nodes, E is actions, result is summaries
- For report think about how to formalize task of summarization in ASG (how thought evolve)
    1. Originally just create summaries from actions
    2. Now preprocess to prune and homogenize, produce and score summaries, take best and fix grammar
- Compare with NN
    1. Randomize action(...) to generate summary(...) on trained ASG
    2. Train NN to generate same summary(...)
    3. Show framework is sane and expandable (computationally tractable)
    4. Compute Rouge score (PyRouge, must clone repo into project) on ASG and NN

- Final goal: take story-specific ASG and general rules to generate summaries, then use top 5/10

## Datasets

https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/
http://nlpprogress.com/english/summarization.html
https://paperswithcode.com/sota/text-summarization-on-gigaword

- SQuAD
- bAbI (very short but need to write summary)
- GigaWord (longer)

## Sentence simplification

- Multiple clauses: split into multiple sentences
- Punctuation
    - [?]: remove clause
    - [!|,|;]: transform into "."
    - [—]: delete inner part

## Representation

http://universalteacher.org.uk/lang/engstruct.htm

Ideas:
- final fix-up using language_checker.fix
- hard-code determiners into derivations (https://www.ef.com/wwen/english-resources/english-grammar/determiners/)
- use lots of simple/precise rules rather than complicated/general ones to minimize ss
- keep rules as restricted as possible, when concept implemented over time add missing rules
- to avoid having to add grammar constraints try and rely on grammar of input

- for learning actions do one sentence at a time to minimize ss
- pick best summary according to TTR*

action(INDEX, VERB, SUBJECT, OBJECT)
summary(VERB, SUBJECT, OBJECT)

verb(INDICATIVE_FORM, TENSE)
subject(NOUN, DET, ADJ_OR_ADV)
object(NOUN, DET, ADJ_OR_ADV)
noun(NAME)
adj_or_adv(NAME)
det(...)
compound(FIRST, SECOND)         # for verbs
conjunct(FIRST, SECOND)         # learn both
disjunct(FIRST, SECOND)         # use choice rule

* Type-Token Ratio (TTR): The basic idea behind that measure is that if the text is more complex, the author uses a more varied vocabulary so there’s a larger number of types (unique words). This logic is explicit in the TTR’s formula, which calculates the number of types divided by the number of tokens. As a result, the higher the TTR, the higher the lexical complexity.