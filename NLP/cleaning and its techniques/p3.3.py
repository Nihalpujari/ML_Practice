import spacy
import nltk
nlp = spacy.load("de_")
text="i am from India and i am indian and in SRH University which has many Unversities"
WORDS=nltk.word_tokenize(text)
doc=[lematizer.lematize(word) in WORDS]
print(doc)