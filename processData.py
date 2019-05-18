import nltk
from nltk import word_tokenize, FreqDist, ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import recall_score

from sklearn.model_selection import GridSearchCV
import numpy as np
import json
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def getData(file_name): # Get data in terms of 'Reviews'
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

def preprocess(file_name, stemmer): # Whether with stemmer or not
    if stemmer: return filter(stemmize(tokenize(getData(file_name))))
    else: return filter(tokenize(getData(file_name)))

def freqMost(file_name, num):
    fdist = FreqDist(preprocess(file_name, False))
    return fdist.most_common(num)

# Create N-gram collection
def listNgrams(filter, num):
    collection = []
    ngram = ngrams(filter, num)
    for grams in ngram: collection.append(grams)
    return collection

# Get most 'limit' N-gram which occur less than 'freq' times
def listFreqBigram(collection, freq, limit):
    fdist = FreqDist(collection)
    listedFreq = [(w,fdist[w]) for w in collection if fdist[w]==freq]
    listedFreq = sorted(set(listedFreq), key=lambda tup: tup[1], reverse=True)
    return listedFreq[:limit]

# Create collection with scores
def scoredBigram(filteredCollection):
    tokens = len(preprocess('reviews.json', False))
    return [(g[0], g[1]/float(tokens)) for g in filteredCollection]

# Get best scored bigrams
def sortedBigram(scoredCollection):
    return [(i[0]) for i in scoredCollection]

# POS-Tag the collection
def posTagger(text):
    return nltk.pos_tag(text)

# Create tags list with number correspending to
def numOfTags(collection):
    tags = {}
    for w in collection:
        if w[1] not in tags: tags[w[1]] = 1
        else: tags[w[1]] += 1
    return tags

# Return the most common words that refers to given pos-tag
def findWords(posList, tag, limit):
    words = []
    for w in posList:
        if(w[1]==tag): words.append(w[0])
    fdist = FreqDist(words)
    return fdist.most_common(limit)

#### PHASE 4 ####
def tokenize_sentences(sentence):
    words = []
    for s in sentence:
        w = extract_words(s)
        words.extend(w)

    words = sorted(list(set(words)))
    return words


def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^\w]", " ", sentence).split()  # nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned


def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1

    return np.array(bag)


sentences = ["At eight o'clock on Thursday morning,Arthur didn't feel very good."]
vocabulary = tokenize_sentences(sentences)
print(bagofwords("At eight o'clock on Thursday morning", vocabulary))




# Load data for train and test
def prepare_data_for_bayes(file_name, train_per):
    # Get data from JSON file
    file = open(file_name)
    data_file = json.load(file)
    file.close()
    # Divide the data into train and test set
    #   in terms of given percentage of train set
    train_data = []
    train_target = []
    test_data = []
    test_target = []
    train_len = int(len(data_file)*train_per/100) # Train set length
    test_len = len(data_file) - train_len # Rest of it is for test
    for i in range(train_len):
        train_data.append(data_file[i]["review_text"])
        train_target.append(data_file[i]["rating"])
    for i in range(test_len):
        test_data.append(data_file[i]["review_text"])
        test_target.append(data_file[i]["rating"])
    return train_data, train_target, test_data, test_target



#  Naive Bayes algorithm
def naive_bayes(data):
    train_data = data[0]
    train_target = data[1]
    test_data = data[2]
    test_target = data[3]
    text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB()),
    ])
    text_clf = text_clf.fit(train_data, train_target) # Train data
    predicted = text_clf.predict(test_data) # Test prediction
    
    return np.mean(predicted == test_target) # Score




# Support Vector Machines(SVM) algorithm
def svm(data):
    train_data = data[0]
    train_target = data[1]
    test_data = data[2]
    test_target = data[3]
    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                         alpha=1e-3, n_iter=5, random_state=42)),
    ])
    text_clf_svm = text_clf_svm.fit(train_data, train_target) # Train data
    predicted_svm = text_clf_svm.predict(test_data) # Test prediction
    return np.mean(predicted_svm == test_target) # Score


#Knn

def knn(data):
    train_data = data[0]
    train_target = data[1]
    test_data = data[2]
    test_target = data[3]
    text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', KNeighborsClassifier()),
    ])
    text_clf = text_clf.fit(train_data, train_target) # Train data
    predicted = text_clf.predict(test_data) # Test prediction
    
    return np.mean(predicted == test_target) # Score


def linearRegression(data):

    train_data = data[0]
    train_target = data[1]
    test_data = data[2]
    test_target = data[3]
    text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LinearRegression()),
    ])
    text_clf = text_clf.fit(train_data, train_target) # Train data
    predicted = text_clf.predict(test_data) # Test prediction
    
    return np.mean(predicted == test_target) # Score


 







### TEST ###

data = prepare_data_for_bayes("reviews.json", 80)


result = naive_bayes(data)
result_svm = svm(data)
result_knn = knn(data)
result_linear = linearRegression(data)

print()
print("---///// TEST /////---")
print("Total reviews classified:\t %i" % len(data[0]))
print("Number of reviews tested:\t %i" % len(data[2]))
print("Score of Naive Bayes:\t\t %f" % result)
print("Score of Support Vector Machines:%f" % result_svm)
print("Knn Score:%f" % result_knn)
print("LinearRegression Score:%f" % result_linear)





