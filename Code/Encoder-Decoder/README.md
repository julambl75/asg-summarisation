# Dataset

The words in words.csv file are the most frequent in English and come from here: http://www.wordfrequency.info/. This contains nouns, adjectives and verbs.

# Libraries

We use a library called [Pattern](\href{http://web.archive.org/web/20190516161631/https://www.clips.uantwerpen.be/pages/pattern-en), which allows us to conjugate verbs, as well as toggle nouns between singular and plural.

We also use the [Datamuse API](\href{https://www.datamuse.com/api/), which lets us find words which are semantically related to a given word in a certain way.

# Definitions

__Holonym__: something is one of its constituents; "lightbulb" is a holonym of "lamp".

__Meronym__: an object which something is part of; "house" is a meronym of "kitchen".

# Stories

For each story, we chose a random noun from the dataset, which we call the topic.

We query from the __Datamuse API__ for an adjective often modified by the topic belonging to the dataset of words. If none are found, then we do not need to use an adjective.

We also ask the __Datamuse API__ to find verbs that are related to the chosen topic. We chose one and this becomes the lexical verb. If it cannot find one then we use the verb "to be".

# Story Types

Using this tool it is possible to generate stories which lead to two types of summaries: conjunctive and descriptive.

## Conjunctive Summaries

These are stories which consist of multiple sentences, all having the story's topic as a subject, as well as sharing the same verb.