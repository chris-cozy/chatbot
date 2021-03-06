import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

#private variables for user file
uregister = "users.json"

#Now we have to create functions. We have all of the data but it is in 0s and 1s, have to convert it into a useable state
#4 functions. Cleaning up sentences, Getting the bag of words, Predicting the class based on the sentence, Getting the response
#LOADS USER DATA
def user_data():
    #opens json file and loads users
    with open(uregister, "r") as f:
        users = json.load(f)
    return users


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    #Sort by probability in reverse order, highest probability first
    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    #return list full of intents and probabilities
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    #added code to test if pieces of code work - successful. many more capabilities.
    if tag == "code test":
        print("test success")

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Jade is now running\n")
name = input("Enter name: ")
users = user_data()
# if name in users, log in, if not, add name to users
# if name in users:
# else:
#     users[name]


while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Jade: " + res + "\n")


#TO DO
# make a user memory system
# nested intents
# #