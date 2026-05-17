from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize 
import string
from matplotlib import pyplot as plt
import numpy as np
reviews = [
    "The product is fantastic! It works like a charm.",
    "I hated the product. It broke after one use.",
    "Product was okay, not the best, but fine overall."
]
# Preprocess the reviews
def preprocess(text):
    text=text.lower()
    tokens=word_tokenize(text)
    clean=[word for word in tokens if text not in string.punctuation]
    return " ".join(clean)
    
cleaned_reviews = [preprocess(review) for review in reviews]

vectorizer = CountVectorizer()
# Fit the vectorizer
vectorizer.fit(cleaned_reviews)
X=vectorizer.transform(cleaned_reviews)

# Print the vocabulary 
words=vectorizer.get_feature_names_out()
count=np.sum(X.toarray(),axis=0)

plt.bar(words,count)
plt.xlabel('Words') 
plt.ylabel('Count')
plt.title('Word Count in Reviews')
plt.xticks(rotation=45)
plt.show()