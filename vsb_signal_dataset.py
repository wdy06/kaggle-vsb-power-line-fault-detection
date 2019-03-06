import argparse
import pandas as pd
import pyarrow.parquet as pq
import os 
import numpy as np
from keras.layers import *
import tensorflow as tf
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from keras import backend as K
from keras import optimizers
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from keras.callbacks import *

import utils


class VsbSignalDataset():
    def __init__(self, mode='train', debug=False):
        self.mode = mode
        if self.mode == 'train':
            self.filename = utils.TRAIN_DATA
            self.metadata = pd.read_csv(utils.TRAIN_META)
        elif self.mode == 'test':
            self.filename = utils.TEST_DATA
            self.metadata = pd.read_csv(utils.TEST_META)
        
        if debug:
          # use first 102 rows in the debug mode
          self.metadata = self.metadata.loc[0:101, :]

    def __getitem__(self, index):
        if isinstance(index, int):
            col_to_load = str(self.metadata['signal_id'].loc[index])
            signal = pq.read_pandas(self.filename, columns=[col_to_load]).to_pandas()
            signal = np.array(signal).reshape(-1).astype(np.float32)
        else:
            if isinstance(index, slice):
                start = index.start if index.start is not None else 0
                stop = index.stop if index.stop is not None else len(self)
                print(stop)
                step = index.step if index.step is not None else 1
                if (start < 0) or (stop < 0) or (step < 0):
                    raise ValueError('start and stop and step must be not minus')
                indices = list(range(start, stop, step))
            elif isinstance(index, list):
                indices = index
            elif isinstance(index, np.ndarray):
                indices = index.tolist()
            else:
                raise ValueError('index must be int ,slice, list or np.ndarray')
            col_to_load = self.metadata['signal_id'].loc[indices]
            col_to_load = list(map(str, list(col_to_load)))
            #print(col_to_load)
            signal = pq.read_pandas(self.filename, columns=col_to_load).to_pandas().astype(np.float32)
            signal = signal.values.T

            
        return signal
    
    def __len__(self):
        return self.metadata.shape[0]
    
    def get_folds(self, n_splits, random_state=42):
        if self.mode != 'train':
            raise ValueError('mode must be train')
        group_df = self.metadata.groupby(['id_measurement']).sum()
        skt = StratifiedKFold(n_splits=n_splits, random_state=random_state)
        folds = list(skt.split(group_df.index, group_df['target']))
        return folds
        
    @property
    def signal_ids(self):
        return self.metadata['signal_id'].values
    @property
    def labels(self):
        if self.mode == 'train':
            return self.metadata['target'].values
        else:
            return None
    
    @property
    def groups(self):
        return self.metadata['id_measurement'].values
    
    def groupid2signalid(self, id_measurement):
        return self.metadata.query('id_measurement==@id_measurement')['signal_id'].values
    