# Before Running

For this to work, you must first [download CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html) from the Stanford website.

Also make sure you have all the required Python3 packages by running `pip3 install -r requirements.txt`.

# How To Run

1. `cd` into the directory where you have `stanford-corenlp-full-2018-10-05`, and run `java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000"`. It is easiest if you create an alias to run this command.
2. Run `main.py`, passing in a string (using `-t`) or path to a text (`-f`) which you want to summarise.