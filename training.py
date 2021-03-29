import random
import json
import pickle
import numpy as np
#natural language toolkit
import nltk
#reduces the word to its stem so its not wasting time looking for the exact word
#for example works work working all seen as the same
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words = []
classes = []
#this holds the pairing of words to their classes
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #this splits up the phrase into a collection of individual words
        word_list = nltk.tokenize(pattern)
        word.append(word_list)
        documents.append((word_list), intent['tag'])