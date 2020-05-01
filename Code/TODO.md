## To talk about

- Expansion of derivations for learning examples
- Expansion of summary rules for learning examples
- Parsing sentence by sentence then combining for ss reduction
- Action creator

## Datasets

https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/
http://nlpprogress.com/english/summarization.html
https://paperswithcode.com/sota/text-summarization-on-gigaword

- SQuAD
- bAbI (very short but need to write summary)
- GigaWord (longer)

## TODO

- Expand language derivations
- Expand summary rules
- Compare with NN
    1. Randomize action(...) to generate summary(...) on trained ASG
    2. Train NN to generate same summary(...)
    3. Show framework is sane and expandable (computationally tractable)

- Start to collect results for report (may take time)
- Think about initial motivation
- Maybe formalize mathematically task of summarization (with CFG, BK, E+, E-)
- For report think about how to formalize task of summarization in ASG (how thought evolve)
- Compute Rouge score (PyRouge, must clone repo into project) on ASG and NN

- Final goal: take story-specific ASG and general rules to generate summaries, then use top 5/10

## Representation

http://universalteacher.org.uk/lang/engstruct.htm

Ideas:
- final fix-up using language_checker.fix
- hard-code determiners into derivations (https://www.ef.com/wwen/english-resources/english-grammar/determiners/)
- use lots of simple/precise rules rather than complicated/general ones to minimize ss
- keep rules as restricted as possible, when concept implemented over time add missing rules
- to avoid having to add grammar constraints try and rely on grammar of input

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