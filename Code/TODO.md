## Ideas for learning

- Recognize start and end of process -> do not talk about intermediate steps

## TODO

- More complicated story/summary pairs (for NN and ASG)

- Take Matthew example and simplify ASG to make work
- 2 ASGs: less strict for learning actions, very strict for generating summaries

- Start to collect results for report (may take time)
- Final goal: take story-specific ASG and general rules to generate summaries, then use top 5/10

## Ideas for representation

https://davehowcroft.com/post/getting-started-with-openccg/
http://www.utcompling.com/wiki/openccg/writing-a-grammar-from-scratch

http://universalteacher.org.uk/lang/engstruct.htm

Ideas:
- final fix-up using language_checker.fix
- hard-code determiners into derivations (https://www.ef.com/wwen/english-resources/english-grammar/determiners/)
- use lots of simple/precise rules rather than complicated/general ones to minimize ss
- keep rules as restricted as possible, when concept implemented over time add missing rules
- to avoid having to add grammar constraints try and rely on grammar of input

action(VERB, SUBJECT, OBJECT)
verb(INDICATIVE_FORM, TENSE)
subject(NOUN, DET, ADJ_OR_ADV)
object(NOUN, DET, ADJ_OR_ADV)
noun(NAME)
adj_or_adv(NAME)
det(...)
conjunct(FIRST, SECOND)
disjunct(FIRST, SECOND)