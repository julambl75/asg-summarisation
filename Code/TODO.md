## To show Alessandra/David

- ss reduction with mode bias (simple example: 396->16, very complicated example: 9477->1044)
- Replace 2 words by superclass (W1 and W2 -> superclass(W1,W2); S V W1 - S V W2 -> S V superclass(W1,W2))
- Increase score if first sentence of summary begins with proper noun
- birdhouse/car


## TODO

- Support numbers as single adj_or_adj object
- Remove comma if between 2 adjectives (recurse with pos=NN/NNS/NNP/NNPS/JJ[, pos]+ CC pos)
- NN
    - Experiments:
        1. ASG rules
        2. Sentences include summary cases from Preprocessor
    - Steps
        1. Improve quality of random stories (lexical fields from WordNet + irrelevant sentences)
        2. Use Preprocessor for generating training data summaries
        3. Train NN

## For report

- Do same as project (main core with acronym like SumASG, then SumASG* to fix/build on top of foundation)
    - Use Peter Little examples
    - Appendix with summary generation rules
    - Talk about expandability (talk about mechanisms)
    - Very nice to not be able to generate grammatically incorrect with ASG
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
    3. Use example of Peter Little to showcase pipeline
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

Choice:
1. Simplify using Python script (faster)
2. Make ASG format more complex (new information probably lost in summary anyway)

- Punctuation
    - [?]: remove clause (helps avoid negation for rhetorical questions)
    - [!|,|;|:]: transform into '.'
    - [–|—]: delete inner part
- Multiple clauses (conjunctive or main+auxiliary): split into multiple sentences
- Adverbs: move to end
- Possessive pronouns and interjections: remove
- Prepositions: remove when at start of sentence
- Contractions: expand
- Acronyms: remove punctuation
- Dependant clauses: split into separate sentence (remove if there is no punctuation)
- Verbless sentences: remove
- Subordinating conjunctions: split into separate sentence
- Preposition clauses: remove if after object
- Complex proper nouns: collapse into CamelCase
- Proper nouns: replace occurrences of pronouns with relevant proper noun (idea: if they are used in the story then there should be little ambiguity)
- Conjunction of common nouns from same lexical field: replace with hypernym (pluralize if items are plural and hypernym plural is used in English)

## Representation

http://universalteacher.org.uk/lang/engstruct.htm

Ideas:
- final fix-up using language_checker.fix
- hard-code determiners into derivations (https://www.ef.com/wwen/english-resources/english-grammar/determiners/)
- use lots of simple/precise rules rather than complicated/general ones to minimize ss
- keep rules as restricted as possible, when concept implemented over time add missing rules
- to avoid having to add grammar constraints try and rely on grammar of input

- reduce search space using mode bias (simple example: 396->16, very complicated example: 9477->1044)
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