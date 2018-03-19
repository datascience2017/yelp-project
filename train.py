import wandb
from wandb.wandb_keras import WandbKerasCallback

from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

import pandas as pd
from sklearn.model_selection import train_test_split

run = wandb.init()
config = run.config
summary = run.summary

df = pd.read_csv('yelp.csv')

text = df['text'].astype(str)
target = df['sentiment']

#import numpy as np
#exit()

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(list(text))
sequences = tokenizer.texts_to_sequences(list(text))
data = pad_sequences(sequences, maxlen=300)

category_to_num = {"Negative": 0, "Positive": 1}
target_num = [category_to_num[t] for t in target]
target_one_hot = np_utils.to_categorical(target_num)

# print(np.unique(target_one_hot))
# exit()

X_train, X_test, y_train, y_test = train_test_split(data, target_one_hot, test_size=0.33, random_state=42)

model = Sequential()
model.add(Embedding(1000, 128, input_length=300))
model.add(Conv1D(32, (5), activation='relu'))
model.add(Dropout(.4))
model.add(MaxPooling1D())
model.add(Conv1D(32, (5), activation='relu'))
model.add(Dropout(.4))
model.add(MaxPooling1D())
model.add(Conv1D(32, (5), activation='relu'))
model.add(Dropout(.4))
model.add(MaxPooling1D())
model.add(Dense(300,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(30,activation='relu'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            callbacks=[WandbKerasCallback()], epochs=10)
