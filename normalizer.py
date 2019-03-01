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

class Normalizer():
    def __init__(self, min_data, max_data, range_needed=(-1,1)):
        self.min_data = min_data
        self.max_data = max_data
        self.range_needed = range_needed
        
    def ts_normalize(self, signal):
        ts_std = (signal - self.min_data) / (self.max_data - self.min_data)
        return ts_std * (self.range_needed[1] - self.range_needed[0]) + self.range_needed[0]
