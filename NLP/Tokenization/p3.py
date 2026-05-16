import nltk
from nltk.corpus import stopwords
import string
punkt=string.punctuation

stop_words=stopwords.words('english')
#print(stop_words)

# output
# ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"]
text="""this is an example of stop words removal in nlp using nltk library in python programming language and this is a very useful technique in text preprocessing for machine learning and natural 
language processing tasks and it helps in improving the performance of the model by reducing the noise in the data"""

WORDS=nltk.word_tokenize(text)
filtered_words=[word for word in WORDS if word.lower() not in stop_words]
print(filtered_words)
# output
# ['example', 'stop', 'words', 'removal', 'nlp', 'using', 'nltk', 'library', 'python', 'programming', 'language', 'useful', 'technique', 'text', 'preprocessing', 'machine', 'learning', 'natural', '``', 'language', 'processing', 'tasks', 'helps', 'improving', 'performance', 'model', 'reducing', 'noise', 'data']

clean=[word for word in filtered_words if word.lower() not in punkt]
print(clean)
# output
# ['example', 'stop', 'words', 'removal', 'nlp', 'using', 'nltk', 'library', 'python', 'programming', 'language', 'useful', 'technique', 'text', 'preprocessing', 'machine', 'learning', 'natural', 'language', 'processing', 'tasks', 'helps', 'improving', 'performance', 'model', 'reducing', 'noise', 'data']