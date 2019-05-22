from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional, CuDNNLSTM
from config_lstm import *
from keras import regularizers

def BiLSTM(num_classes, reg, reg_wt):

    data_dim = 4096
    
    model = Sequential()
    
    if reg==False:
        model.add(Bidirectional(CuDNNLSTM(1024, return_sequences=True), input_shape=(None, data_dim)))
        model.add(Bidirectional(CuDNNLSTM(512, return_sequences=True)))
        model.add(Bidirectional(CuDNNLSTM(256, return_sequences=False)))
    
    else:
        
        model.add(Bidirectional(CuDNNLSTM(1024, return_sequences=True, kernel_regularizer=regularizers.l1_l2(reg_wt), recurrent_regularizer=regularizers.l1_l2(reg_wt),
                                                kernel_initializer='he_normal', recurrent_initializer='he_normal'), input_shape=(None, data_dim)))
        
        model.add(Bidirectional(CuDNNLSTM(512, return_sequences=True, kernel_regularizer=regularizers.l1_l2(reg_wt), recurrent_regularizer=regularizers.l1_l2(reg_wt),
                                                kernel_initializer='he_normal', recurrent_initializer='he_normal')))

        model.add(Bidirectional(CuDNNLSTM(256, return_sequences=False, kernel_regularizer=regularizers.l1_l2(reg_wt), recurrent_regularizer=regularizers.l1_l2(reg_wt),
                                                kernel_initializer='he_normal', recurrent_initializer='he_normal')))

    
    model.add(Dropout(DROPOUT_RATE, seed=1))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(DROPOUT_RATE, seed=1))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='sigmoid', kernel_initializer='he_normal'))

    return model

if __name__ == '__main__':
    model = BiLSTM(num_classes)
    print(model.summary())