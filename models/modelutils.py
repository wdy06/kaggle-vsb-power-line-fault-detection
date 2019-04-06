
import numpy as np

import keras
import keras.backend as K
from keras.layers import *
from keras.models import Sequential, Model

from .attention import Attention

def get_model(modelname, input_shape=None, n_outputs=1, lstm_units=64, dense_units=64):
    if modelname == 'cnn_lstm':
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=16, kernel_size=32, strides=2, activation='relu'), input_shape=(None,input_shape[2], input_shape[3])))
        model.add(TimeDistributed(Conv1D(filters=8, kernel_size=16, strides=2, activation='relu')))
        model.add(TimeDistributed(Conv1D(filters=4, kernel_size=4, strides=2, activation='relu')))
#         model.add(TimeDistributed(Dropout(0.5)))
#         model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(CuDNNLSTM(32))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))
        
#     elif modelname == 'wavenet':
#         model = Sequential()
#         model.add(AtrousConvolution1D(filters=16, kernel_size=2, atrous_rate=1, border_mode='same', activation='relu'), 
#                                       input_shape=(None,input_shape[2], input_shape[3]))
#         model.add(AtrousConvolution1D(filters=16, kernel_size=2, atrous_rate=2, 
#                                       border_mode='same', activation='relu')
#         model.add(AtrousConvolution1D(filters=16, kernel_size=2, atrous_rate=4, 
#                                       border_mode='same', activation='relu')
         
#         model.add(Dropout(0.5))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dense(n_outputs, activation='sigmoid'))
    
    elif modelname == 'bidirectional_lstm':
        if input_shape is None:
            raise ValueError('input shape is needed.')
        
        inp = Input(shape=(input_shape[1], input_shape[2],))
        x = Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True))(inp)
        x = Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True))(x)
        x = Attention(input_shape[1])(x)
        x = Dense(dense_units, activation="relu")(x)
        x = Dense(n_outputs, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        
    else:
        raise ValueError('unknown model name.')
        
    model.summary()
    return model
            
            
            
            
            
            
            

            