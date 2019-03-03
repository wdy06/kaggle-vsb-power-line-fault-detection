import pandas as pd
import numpy as np
import feather
import dask.array as da
import pyarrow.parquet as pq

import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf
import gc
from numba import jit
from IPython.display import display, clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold

import utils
from vsb_signal_dataset import VsbSignalDataset


@jit('float32(float32[:], int32, int32)')
def window_extracter(signal, window_size, stride):
    feature = []
    for start in range(0, len(signal), stride):
        #print(start)
        if len(signal) < start+window_size:
            break
        ts_range = signal[start:start+window_size]
        feature.append(ts_range)
    return feature

@jit('float32(float32[:], int32, int32)')
def time_feature_extracter(signal, window_size, stride):
    feature = []
    for start in range(0, len(signal), stride):
        if len(signal) < start+window_size:
            break
        ts_range = signal[start:start+window_size]
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std() # standard deviation
        std_top = mean + std # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100]) 
        max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        feature.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),percentil_calc, relative_percentile]))
    return feature


def feature_extracter(extracter, dataset, window_size, stride, grouped=False, normalizer=None):
    divide_num = 12
    chunk_size = (len(dataset) // divide_num) * 3
    groups = np.unique(dataset.groups)
    X = []
    for start_index in tqdm(range(0, len(dataset), chunk_size)):
        
        if start_index+chunk_size<=len(dataset):
            signals = dataset[start_index:start_index+chunk_size]
        else:
            signals = dataset[start_index:]
        for i in range(int(len(signals)/3)):
            grouped_X = []
            for phase in [0, 1, 2]:
                sig = signals[i*3+phase]
                if normalizer is not None:
                    feature = extracter(normalizer.ts_normalize(sig), window_size, stride)
                else:
                    feature = extracter(sig, window_size, stride)
                if grouped:
                    grouped_X.append(feature)
                else:
                    X.append(feature)
            if grouped:
                grouped_X = np.concatenate(grouped_X, axis=1)
                X.append(grouped_X)
    return np.array(X)

# def feature_extracter(extracter, dataset, window_size, stride, grouped=False, normalizer=None):
#     divide_num = 6
#     X = []
#     for indices in tqdm(np.array_split(np.arange(len(dataset)), divide_num)):
#         signals = dataset[indices]
#         for sig in signals:
#             if normalizer:
#                 sig = normalizer.ts_normalize(sig)
#             feature = extracter(sig, window_size, stride)
#             X.append(feature)
#     return np.array(X)
        











