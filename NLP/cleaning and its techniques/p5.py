import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

review = "I have been FLYING a lot lately and the Flights just keep getting DELAYED. Honestly, traveling for WORK gets exhausting with endless delays, but every trip teaches you something new!"

# Lowercase the review
lower_text = review.lower()

# Tokenize the lower_text into words
tokens = word_tokenize(lower_text)

# Remove stop words and punctuation
stop_words = stopwords.words('english')

clean_tokens = [word for word in tokens 
                if word not in stop_words 
                and word not in string.punctuation]

print(clean_tokens)