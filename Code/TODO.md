## TODO

- refactor GenActions (now use same subject and verb per story, no more pronouns or names; remove bad code)
- idea: try test data combining both structures
- use smaller max for Datamuse
- pick longest top summary

S V O1. S V O2. S V O3.
S V O. it was A O

the [chief] [engineer] [organized] a [number] . it was a [large] [number] . 

## For Report

To write: smaller hidden size and very slow training force learning to be more general, no more names

Params: dropout 0.25; lr 0.001; batchsize 50; wordvecsize 500; hiddensize 500
Same nothing: final validation accuracy 53%
Same subject: final validation accuracy 64%

## Datasets

https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/
http://nlpprogress.com/english/summarization.html
https://paperswithcode.com/sota/text-summarization-on-gigaword

- SQuAD
- bAbI (very short but need to write summary)
- GigaWord (longer)