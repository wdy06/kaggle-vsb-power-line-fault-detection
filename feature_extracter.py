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
import os
from scipy.signal import argrelmax, find_peaks, peak_widths, peak_prominences
from scipy.stats import kurtosis, entropy, skew


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

@jit('float32(float32[:], int32, int32)')
def peak_feature_extracter(signal, window_size, stride):
    feature = []
    for start in range(0, len(signal), stride):
        if len(signal) < start+window_size:
            break
        ts_range = signal[start:start+window_size]
        # calculate peak feature
        peaks, _ = find_peaks(ts_range, threshold=0.05)
        if len(peaks) == 0:
            num_of_peaks = 0.
            min_width_of_peak = 0.
            max_width_of_peak = 0.
            mean_width_of_peak = 0.
            std_width_of_peak = 0.

            min_height_of_peak = 0.
            max_height_of_peak = 0.
            mean_height_of_peak = 0.
            std_height_of_peak = 0.

            min_prominence = 0.
            max_prominence = 0.
            mean_prominence = 0.
            std_prominence = 0.
        else:
            widths = peak_widths(ts_range, peaks, rel_height=0.5)
            num_of_peaks = len(peaks)/len(ts_range)
            min_width_of_peak = np.min(widths[0])/100
            max_width_of_peak = np.max(widths[0])/100
            mean_width_of_peak = np.mean(widths[0])/100
            #std_width_of_peak = np.std(widths[0])/100

            min_height_of_peak = np.min(widths[1])
            max_height_of_peak = np.max(widths[1])
            mean_height_of_peak = np.mean(widths[1])
            #std_height_of_peak = np.std(widths[1])

            prominences = peak_prominences(ts_range, peaks)[0]
            min_prominence = np.min(prominences)
            max_prominence = np.max(prominences)
            mean_prominence = np.mean(prominences)
            #std_prominence = np.std(prominences)
        
        
        feature.append(np.asarray([num_of_peaks, 
                                   min_width_of_peak, 
                                   max_width_of_peak, 
                                   mean_width_of_peak,
                                   #std_width_of_peak, 
                                   min_height_of_peak, 
                                   max_height_of_peak, 
                                   mean_height_of_peak, 
                                   #std_height_of_peak, 
                                   min_prominence,
                                   max_prominence, 
                                   mean_prominence
                                   #std_prominence
                                  ]))
    return feature

@jit('float32(float32[:], int32, int32)')
def adv_stats_feature_extracter(signal, window_size, stride):
    feature = []
    for start in range(0, len(signal), stride):
        if len(signal) < start+window_size:
            break
        ts_range = signal[start:start+window_size]
        # calculate each feature
        feature.append(np.asarray([kurtosis(ts_range), skew(ts_range)]))
    return feature

feature_funcs = {
    'stats': time_feature_extracter,
    'window': window_extracter,
    'peak': peak_feature_extracter,
    'adv_stats': adv_stats_feature_extracter
}

def feature_extracter(feature_list, dataset, window_size, stride, grouped=False, normalizer=None, 
                      use_cache=True, save_result=True):
    print(f'feature list: {feature_list}')
    _feature_string = '_'.join(feature_list)
    if grouped:
        cache_path \
            = f'./features/X_{dataset.mode}_group_{_feature_string}_w{window_size}_s{stride}.npy'
    else:
        cache_path \
            = f'./features/X_{dataset.mode}_{_feature_string}_w{window_size}_s{stride}.npy'
        
    if use_cache:
        print('use feature cache')
        if os.path.exists(cache_path):
            print('cache file found !!')
            return np.load(cache_path)
        else:
            print('cache file cannot be find..')
            
    print('extracting feature ...')
    divide_num = 12
    chunk_size = (len(dataset) // divide_num) * 3
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
                if normalizer:
                    sig = normalizer.ts_normalize(sig)
                
                feature_X = []
                for feature_name in feature_list:
                    extracter = feature_funcs[feature_name]
                    feature = extracter(sig, window_size, stride)
                    feature_X.append(feature)
                feature_X = np.concatenate(feature_X, axis=1)
                if grouped:
                    grouped_X.append(feature_X)
                else:
                    X.append(feature_X)
            if grouped:
                grouped_X = np.concatenate(grouped_X, axis=1)
                X.append(grouped_X)
    if save_result:
        print('save features!')
        np.save(cache_path, X)
        
    
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
        











