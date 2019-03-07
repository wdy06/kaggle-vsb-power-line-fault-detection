
import numpy as np

import keras
import keras.backend as K
from keras.layers import *
from keras.models import Sequential, Model

from .attention import Attention

def get_model(modelname, input_shape=None, n_outputs=1):
    if modelname == 'cnn_lstm':
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,input_shape[1],input_shape[2])))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(CuDNNLSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))
    
    elif modelname == 'bidirectional_lstm':
        if input_shape is None:
            raise ValueError('input shape is needed.')
        
        inp = Input(shape=(input_shape[1], input_shape[2],))
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
        x = Attention(input_shape[1])(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(n_outputs, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        
    else:
        raise ValueError('unknown model name.')
        
    return model
            
            
            
            
            
            
            

            