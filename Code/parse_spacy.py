import spacy
from spacy import displacy
import sys

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
if len(sys.argv) > 1:
  try:
    text = open(sys.argv[1]).read()
  except IOError:
    text = sys.argv[1]
else:
  text = "Please pass a filename or string as an argument."
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

displacy.serve(doc, style="dep")
