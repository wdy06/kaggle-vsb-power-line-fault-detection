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
import torch

import utils
from models import modelutils

# fix random seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
import random as rn
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='atlas-protein-image-classification on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

# select how many folds will be created

N_SPLITS = 5
epoch = 50
if args.debug:
    N_SPLITS = 2
    epoch = 5
    
batchsize = 128
# it is just a constant with the measurements data size
sample_size = 800000
# in other notebook I have extracted the min and max values from the train data, the measurements
max_num = 127
min_num = -128
n_dim = 160

# just load train data
df_train = pd.read_csv(utils.TRAIN_META)
# set index, it makes the data access much faster
df_train = df_train.set_index(['id_measurement', 'phase'])


# this code is very simple, divide the total size of the df_train into two sets and process it
X = []
y = []

total_size = len(df_train)
#for ini, end in [(0, int(total_size/2)), (int(total_size/2), total_size)]:
#X_temp, y_temp = utils.prep_data(df_train, 0, tatal_size, n_dim, min_num, max_num, sample_size)
X, y = utils.prep_data(df_train, 0, total_size, n_dim, min_num, max_num, sample_size)
# X.append(X_temp)
# y.append(y_temp)
    
# X = np.concatenate(X)
# y = np.concatenate(y)


print(X.shape, y.shape)
np.save("X.npy",X)
np.save("y.npy",y)

# Here is where the training happens

# First, create a set of indexes of the 5 folds
splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019).split(X, y))
preds_val = []
y_val = []
# Then, iteract with each fold
# If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]
for idx, (train_idx, val_idx) in enumerate(splits):
    K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)
    print("Beginning fold {}".format(idx+1))
    # use the indexes to extract the folds in the train and validation data
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    # instantiate the model for this fold
    model = modelutils.get_model('bidirectional_lstm', train_X.shape)
    # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
    # validation matthews_correlation greater than the last one.
    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[utils.matthews_correlation])
    # Train, train, train
    model.fit(train_X, train_y, batch_size=batchsize, epochs=epoch, validation_data=[val_X, val_y], callbacks=[ckpt])
    # loads the best weights saved by the checkpoint
    model.load_weights('weights_{}.h5'.format(idx))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict(val_X, batch_size=512))
    # and the val true y
    y_val.append(val_y)

# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[...,0]
y_val = np.concatenate(y_val)

best_sarch_result = utils.threshold_search(y_val, preds_val)
best_threshold = best_sarch_result['threshold']
best_val_score = best_sarch_result['matthews_correlation']

print(f'best validation score: {best_val_score}')

meta_test = pd.read_csv('./data/metadata_test.csv')
meta_test = meta_test.set_index(['signal_id'])

# First we daclarete a series of parameters to initiate the loading of the main data
# it is too large, it is impossible to load in one time, so we are doing it in dividing in 10 parts
first_sig = meta_test.index[0]
n_parts = 10
max_line = len(meta_test)
part_size = int(max_line / n_parts)
last_part = max_line % n_parts
print(first_sig, n_parts, max_line, part_size, last_part, n_parts * part_size + last_part)
# Here we create a list of lists with start index and end index for each of the 10 parts and one for the last partial part
start_end = [[x, x+part_size] for x in range(first_sig, max_line + first_sig, part_size)]
start_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]
print(start_end)
X_test = []
# now, very like we did above with the train data, we convert the test data part by part
# transforming the 3 phases 800000 measurement in matrix (160,57)
for start, end in start_end:
    subset_test = pq.read_pandas('./data/test.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    for i in tqdm(subset_test.columns):
        id_measurement, phase = meta_test.loc[int(i)]
        subset_test_col = subset_test[i]
        subset_trans = utils.transform_ts(subset_test_col, sample_size, n_dim, min_num, max_num)
        X_test.append([i, id_measurement, phase, subset_trans])


X_test_input = np.asarray([np.concatenate([X_test[i][3],X_test[i+1][3], X_test[i+2][3]], axis=1) for i in range(0,len(X_test), 3)])
np.save("X_test.npy",X_test_input)


submission = pd.read_csv('./data/sample_submission.csv')
print(len(submission))

preds_test = []
for i in range(N_SPLITS):
    model.load_weights('weights_{}.h5'.format(i))
    pred = model.predict(X_test_input, batch_size=300, verbose=1)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(3):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)

preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)

submission['target'] = preds_test
submission_filename = f'submission_val{best_val_score:.4f}.csv'
submission.to_csv(submission_filename, index=False)
print(f'save result to {submission_filename}')





