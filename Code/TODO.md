Changes: lr, fix word embeddings, wordvecsize, hiddensize
To write: smaller hidden size forces learning to be more general


1. dropout 0.25; lr 0.001; batchsize 50; wordvecsize 500; hiddensize 500
2. dropout 0.25; lr 0.0002; batchsize 50; wordvecsize 100; hiddensize 50
3. dropout 0.25; lr 0.0005; batchsize 50; wordvecsize 100; hiddensize 50
3. dropout 0.25; lr 0.001; batchsize 50; wordvecsize 200; hiddensize 50

## Datasets

https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/
http://nlpprogress.com/english/summarization.html
https://paperswithcode.com/sota/text-summarization-on-gigaword

- SQuAD
- bAbI (very short but need to write summary)
- GigaWord (longer)