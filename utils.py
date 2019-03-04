import feather
import pandas as pd
import numpy as np
import os
import json
import pyarrow.parquet as pq
from tqdm import tqdm

import keras.backend as K


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_META = os.path.join(DIR_PATH, 'data/metadata_train.csv')
TEST_META = os.path.join(DIR_PATH, 'data/metadata_test.csv')
TRAIN_DATA = os.path.join(DIR_PATH, 'data/train.parquet')
TEST_DATA= os.path.join(DIR_PATH, 'data/test.parquet')
RESULT_DIR = os.path.join(DIR_PATH, 'result')


def load_train_meta():
    file_path = os.path.join(DIR_PATH, 'data/metadata_train.fth')
    return feather.read_dataframe(file_path)

def load_test_meta():
    file_path = os.path.join(DIR_PATH, 'data/metadata_test.fth')
    return feather.read_dataframe(file_path)


def load_train_data():
    file_path = os.path.join(DIR_PATH, 'data/train.fth')
    return feather.read_dataframe(file_path)

def load_test_data():
    file_path = os.path.join(DIR_PATH, 'data/test.fth')
    return feather.read_dataframe(file_path)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f)
    
def group_stratified_kfold(meta_df, n_splits=5, random_state=42):
    group_df = meta_df.groupby(['id_measurement']).sum()
    skt = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    folds = list(skt.split(group_df.index, group_df['target']))
    return folds

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        #score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64)))
        score = K.eval(matthews_correlation(K.variable(y_true.astype(np.float64)), K.variable((y_proba > threshold).astype(np.float64))))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result


def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    # This function standardize the data from (-128 to 127) to (-1 to 1)
    # Theoretically it helps in the NN Model training, but I didn't tested without it
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:    
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def min_max_normalize(ts_array, min_data, max_data, range_needed=(-1,1)):
    ts_array = (ts_array - min_data) / (max_data - min_data)
    ts_array = ts_array * (range_needed[1] - range_needed[0]) + range_needed[0]
    return ts_array
    
def feature_extracter(ts_array, window_size, stride):
    x = []
    for i in tqdm(range(ts_array.shape[1])):
        ts = ts_array[:,i]
        #print(ts.shape)
        x_tmp = []
        for c in range(0, len(ts_array), stride):
            #print(f'c:{c}')
            x_tmp.append(ts[c:c+window_size])
        x.append(x_tmp)
    
    x = np.array(x)
    
def transform_ts(ts, sample_size, n_dim, min_num, max_num, min_max=(-1,1)):
    # This is one of the most important peace of code of this Kernel
    # Any power line contain 3 phases of 800000 measurements, or 2.4 millions data 
    # It would be praticaly impossible to build a NN with an input of that size
    # The ideia here is to reduce it each phase to a matrix of <n_dim> bins by n features
    # Each bean is a set of 5000 measurements (800000 / 160), so the features are extracted from this 5000 chunk data.
    # convert data into -1 to 1
    ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    # bucket or chunk size, 5000 in this case (800000 / 160)
    bucket_size = int(sample_size / n_dim)
    # new_ts will be the container of the new data
    new_ts = []
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, sample_size, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts_std[i:i + bucket_size]
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
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),percentil_calc, relative_percentile]))
    return np.asarray(new_ts)

def prep_data(df, start, end, n_dim, min_num, max_num, sample_size):
    # this function take a piece of data and convert using transform_ts(), but it does to each of the 3 phases
    # if we would try to do in one time, could exceed the RAM Memmory
    # load a piece of data from file
    praq_train = pq.read_pandas(TRAIN_DATA, columns=[str(i) for i in range(start, end)]).to_pandas()
    X = []
    y = []
    # using tdqm to evaluate processing time
    # takes each index from df_train and iteract it from start to end
    # it is divided by 3 because for each id_measurement there are 3 id_signal, and the start/end parameters are id_signal
    for id_measurement in tqdm(df.index.levels[0].unique()[int(start/3):int(end/3)]):
        X_signal = []
        # for each phase of the signal
        for phase in [0,1,2]:
            signal_id, target = df.loc[id_measurement].loc[phase]
            # but just append the target one time, to not triplicate it
            if phase == 0:
                y.append(target)
            # extract and transform data into sets of features
            X_signal.append(transform_ts(praq_train[str(signal_id)], sample_size=sample_size, n_dim=n_dim, min_num=min_num, max_num=max_num))
        X_signal = np.concatenate(X_signal, axis=1)
        X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y












