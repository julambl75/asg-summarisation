Changes: fix word embeddings, wordvecsize, hiddensize
To write: smaller hidden size and very slow training force learning to be more general

1. dropout 0.25; lr 0.001; batchsize 50; wordvecsize 500; hiddensize 500
2. dropout 0.25; lr 0.0002; batchsize 50; wordvecsize 100; hiddensize 50
3. dropout 0.25; lr 0.0005; batchsize 50; wordvecsize 100; hiddensize 50
4. dropout 0.25; lr 0.001; batchsize 50; wordvecsize 200; hiddensize 50
5. dropout 0.25; lr 0.001; batchsize 50; wordvecsize 200; hiddensize 100
6. dropout 0.25; lr 0.001; batchsize 50; wordvecsize 200; hiddensize 1000
7. dropout 0.25; lr 0.001; batchsize 50; wordvecsize 500; hiddensize 500

Ideas: train first on super consistent data then on normal data

## Datasets

https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/
http://nlpprogress.com/english/summarization.html
https://paperswithcode.com/sota/text-summarization-on-gigaword

- SQuAD
- bAbI (very short but need to write summary)
- GigaWord (longer)