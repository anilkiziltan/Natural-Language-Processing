import nltk
from nltk.corpus import brown
from nltk import word_tokenize, FreqDist, ngrams, pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import json

nltk.download('punkt')
nltk.download('brown')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')



print("Part 1:")
temp = {}
for word, pos in sorted(key for key in brown.tagged_words()[:1000]):
    if word not in temp:
        temp[word] = pos

for x in temp:
    print(f'{x:<20} {temp[x]}')

print("Part 2:")
grammar = nltk.CFG.fromstring("""
 S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)

example_sentence = ['the','dog','saw','a','man','in','the','park']

parser = nltk.ChartParser(grammar)
for tree in parser.parse(example_sentence):
    print(tree)
