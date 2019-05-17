from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional
from config_lstm import *

def BiLSTM(num_classes):

    data_dim = 4096
    
    model = Sequential()
    model.add(Bidirectional(LSTM(2048, return_sequences=True), input_shape=(None, data_dim)))
    model.add(Bidirectional(LSTM(1024, return_sequences=True)))
    model.add(Bidirectional(LSTM(512, return_sequences=False)))
    model.add(Dropout(DROPOUT_RATE, seed=1))
    model.add(Dense(500, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(DROPOUT_RATE, seed=1))
    model.add(Dense(400, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal'))

    return model

if __name__ == '__main__':
    model = BiLSTM(num_classes)
    print(model.summary())