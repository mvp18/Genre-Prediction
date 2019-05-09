
import numpy as np
import os, sys
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Bidirectional
from keras.layers import CuDNNLSTM as LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
import keras.backend as K
import cPickle as pickle

hidden_size = 64

input = Input(shape=input_s, name='input', dtype='float')
lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))(input)
dense = Dense(500, activation='relu')(lstm)
predictions = Dense(11, activation='sigmoid')(dense)

model = Model(input, predicates)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train, Y_train = pickle.load(open('MovieSummeries_train.pickle'))

model.fit(X_train, Y_train, batch_size=32, epochs=50)

X_test, Y_test = pickle.load(open('MovieSummeries_test.pickle'))

new_model = Model(input, dense)

new_model.predict(X_train)
new_model.predict(X_test)
