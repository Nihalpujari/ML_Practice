import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
stemmer=PorterStemmer()
lemmitazer=WordNetLemmatizer()

words=["running","ran","runs","easily","fairly", "bats","organizations"]
stemmed=[stemmer.stem(word) for word in words]
lemmitazed=[lemmitazer.lemmatize(word) for word in words]

print("here is stemmed words",stemmed)
print("here is lemmatized words",lemmitazed)