import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import json
nltk.download('punkt')
nltk.download('stopwords')

# Get data from json file
file_name = "reviews.json"
with open(file_name) as data_file:
    data = json.load(data_file)
text = ""
for i in data:
    text += (i["review_text"])


# Tokenize words without punctuation
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)
#print(tokens)

# Tokenize words with punctuation
# tokens = nltk.word_tokenize(text)

# Get stems of all tokens
stemmer = PorterStemmer()
words = [stemmer.stem(w) for w in tokens]

# Initialize English stopwords
stopWords = set(stopwords.words('english'))
filteredWords = []

# Filter the words with stopwords
for w in words:
    if w not in stopWords:
        filteredWords.append(w)

#5. Display the frequency distribution information of the stemmed text.
fdist = FreqDist(filteredWords)
print(fdist)

#6. Display the most frequent 10 stems.
print(fdist.most_common(10))

#7. Visualize the frequency distribution using graphical plots.

fd = nltk.FreqDist(filteredWords)
fd.plot(30,cumulative = False)


#8. List all the words from the text which have more than 10 letters.
long_words = [w for w in filteredWords if len(w) > 10]
print(sorted(long_words))



