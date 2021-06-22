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

#Calls constructor of the lemmatizer
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
#this holds the pairing of words to their classes
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #this splits up the phrase into a collection of individual words
        word_list = nltk.word_tokenize(pattern)
        #takes the word and appends it to the list
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lemmatizes the words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
#set removes duplicates, sorted sorts the words
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Getting into the Machine Learning part
#We now have the words needed but they are not numerical values. Nueral networks cannot be feed words, it must be numerical values
#Going to use 'bag of words'. Set the individual word values to either 0 or 1 depending on if it's occuring in that particular pattern.
#Doing the same for the classes

training = []
#template of 0s, as many as there are classes
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #for each word if word intents we want to know if it occurs in the pattern. If so signify a 1
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    #this is what we do to copy the list. not type casting but copying
    output_row = list(output_empty)
    #setting this index in the output row to 1
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

#shuffle training data
random.shuffle(training)
#turn it into a np array
training = np.array(training)
#converting into x and y values
train_x = list(training[:, 0])
train_y = list(training[:, 1])

#Onto the MACHINE LEARNING portion
#building the nueral network model, a simple sequential model
model = Sequential()
#adding a couple layers
#input layer, 128 nuerons, input shape that is dependent on training data size
#activation function is a rectified linear unit, or 'relu'
model.add(Dense(128, input_shape = (len(train_x[0]),), activation='relu'))
#All the details of the layers not explained, seperate research: Nueral Network Theory
model.add(Dropout(0.5))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.5))
#Want to have as many nuerons as there are classes, activation function a soft max function, which is the function that will allow us to add up the results
#Scales the results in the output layer so that they all add up to 1
model.add(Dense(len(train_y[0]), activation='softmax'))
#lr is the learning rate
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
#compiles them all
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#epochs is how many times you are feeding the data into the nueral network
hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbotmodel.h5', hist)
print("Trained.")

#After this the nueral network is trained. The next step is creating the chatbot application that uses the trained model
#The more training data, patterns, and intents added, the better and more fluid the recognition will be.