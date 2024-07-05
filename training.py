import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes =[]
documents = []
ignore_letters = ['?', '!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

words =[lemmatizer.lemmatize(word) for word in words if word not in  ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training,dtype=object)

train_x = list(training[:,0])
train_y = list(training[:,1])

mode1 = Sequential()
mode1.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))

mode1.add(Dropout(0,5))
mode1.add(Dense(64, activation='relu'))
mode1.add(Dropout(0,5))
mode1.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
mode1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

mode1.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=2)
mode1.save('chatbot_model.keras')

print("Done")



print(words)



