import nltk
from nltk import word_tokenize, FreqDist, ngrams, pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def getData(file_name):
    with open(file_name) as data_file: data = json.load(data_file)
    text = ''
    for i in data: text += (i["review_text"])
    return text


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens


def stemmize(tokens):
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in tokens]
    return words


def filter(words):
    stopWords = set(stopwords.words('english'))
    filteredWords = []
    for w in words:
        if w not in stopWords: filteredWords.append(w)
    return filteredWords


# 1. preprocess, which takes the text as parameter,and returns the tokenized version of the text that does not contain neither any stop words nor any punctuations.
def preprocess(file_name, stemmer):  # Whether with stemmer or not
    if stemmer:
        return filter(stemmize(tokenize(getData(file_name))))
    else:
        return filter(tokenize(getData(file_name)))


# print(preprocess('reviews.json', False))

# 2. mostFrequent, which takes tokenized version of text and a number n as parameters, and returns the number of the occurrences of the frequent words.
def freqMost(file_name, num):
    fdist = FreqDist(preprocess(file_name, False))
    return fdist.most_common(num)


#print(freqMost('reviews.json', 5))

# 3. displayNgrams, which takes tokenized text and a number n as parameters, and displays n grams only as many as the desired n.
def listNgrams(filter, num):
    collection = []
    ngram = ngrams(filter, num)
    for grams in ngram: collection.append(grams)
    return collection


collection = listNgrams(preprocess('reviews.json', False), 2)

#print(collection)


# 4. mostFreqBigram, which takes frequency of the bigram, number of the bigrams that are going to be listed and a list of bigrams, and returns only the bigrams with the given frequency rate.
def listFreqBigram(collection, freq, limit):
    fdist = FreqDist(collection)
    listedFreq = [(w, fdist[w]) for w in collection if fdist[w] < freq]
    listedFreq = sorted(set(listedFreq), key=lambda tup: tup[1], reverse=True)
    return listedFreq[:limit]


#print(listFreqBigram(collection, 4, 1))

# 6. Create a function that returns the score information of the bigrams that are equal to or more frequent than 2
def scoredBigram(filteredCollection):
    tokens = len(preprocess('reviews.json', False))
    return [(g[0], g[1] / float(tokens)) for g in filteredCollection]


# print(scoredBigram(bigrams))

# 5. Collocations help us find out that which pairs of words are more probable to occur. You are expected to write a function, which takes bigrams as parameters, and returns the top 10 bigrams.
def sortedBigram(scoredCollection):
    return [(i[0]) for i in scoredCollection]


bigrams = listFreqBigram(collection, 100, 10)

#print(sortedBigram(scoredBigram(bigrams)))

# 7. You are expected to create a function that produces a list of words. Each word will have speech tag along with them. You need to make use of POS-taggers.
def get_pos(string):
    string = tokenize(getData(string))
    pos_string = pos_tag(string)
    return pos_string


#print(get_pos('reviews.json'))

# 8. Write a function numOfTags that takes tagged list and returns only the most common tags
def getMostCommonTags(string):
    mostCommonList = list()
    tagList = list()
    for tag in (string):
        tagList.append(tag[1])

    mostTags = FreqDist(tagList).most_common(5)
    mostCommonList.append(mostTags)
    # print(mostTags)
    return mostTags


#print(getMostCommonTags(get_pos('reviews.json')))


# print(getMostCommonTags(get_pos('reviews.json')))
# 9. Write a function, which takes two parameters (one of them for tagged text, and the other one is for a tag)

def get_specified_tag(file_name, tag_prefix):
    define_text = list()
    for a, b in get_pos(file_name):
        if b.startswith(tag_prefix):
            define_text.append(a)
    return define_text


print(get_specified_tag('reviews.json', 'NN'))


# 10. You are expected to list all the words with number of occurrences information, frequency information and rank information along with their speech tags

def frequency_information(tokenize_text):
    stem_words = list()
    for x in tokenize_text:
        stem_words.append(PorterStemmer().stem(x))
    return FreqDist(stem_words)

print(frequency_information('reviews.json'))

#most 3 common words of comments are 'oh, like, well, get and know'. All of them almost said 400 times. We can say that people mostly like the foods, and make posive comments.

# The word 'yelling' is used with the words burned, fired and marge. We can observe that people don't like their foods burned or oiled.

# Top 5 bigram is [(u'mm', u'hmm'), (u'whoo', u'hoo'), (u'uh', u'oh'), (u'like', u'oh'), (u'uh', u'huh'). We can say that people mostly like foods and make comments like mm,hmm,uh,ohh

#most 3 common tags are NN, JJ and PRP which means that people describes their comments about the restaurants with nouns and adjectives instead of verbs.







