import argparse
from datetime import datetime
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
from vsb_signal_dataset import VsbSignalDataset
from normalizer import Normalizer
import feature_extracter
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


parser = argparse.ArgumentParser(description='vsb-power-line-fault-detection on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

if args.debug:
    result_dir = os.path.join(utils.RESULT_DIR, 'debug-'+datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
else:
    result_dir = os.path.join(utils.RESULT_DIR, datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
os.mkdir(result_dir)
print(f'created: {result_dir}')

# select how many folds will be created
N_SPLITS = 5
epoch = 50
if args.debug:
    print('running debug mode')
    epoch = 5
    
batchsize = 128
# it is just a constant with the measurements data size
sample_size = 800000
# in other notebook I have extracted the min and max values from the train data, the measurements
max_num = 127
min_num = -128
#n_dim = 160
window_size = 5000
stride = 5000

dataset = VsbSignalDataset(mode='train', debug=args.debug)
normalizer = Normalizer(min_num, max_num)

X = feature_extracter.feature_extracter(feature_extracter.time_feature_extracter,
                                        dataset, window_size=window_size, stride=stride, grouped=True, 
                                        normalizer=normalizer)

y = dataset.labels
#y = y[::3].reshape(-1, 1)
y = y[::3]

print(X.shape, y.shape)
np.save(os.path.join(result_dir, "X.npy"),X)
np.save(os.path.join(result_dir, "y.npy"),X)

# Here is where the training happens

splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019).split(X, y))
#splits = dataset.get_folds(n_splits=N_SPLITS)

preds_val = []
y_val = []

for idx, (train_idx, val_idx) in enumerate(splits):
    K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)
    print("Beginning fold {}".format(idx+1))
    # use the indexes to extract the folds in the train and validation data
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    # instantiate the model for this fold
    model = modelutils.get_model('bidirectional_lstm', train_X.shape)
    # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
    # validation matthews_correlation greater than the last one.
    ckpt = ModelCheckpoint(os.path.join(result_dir, f'weights_{idx}.h5'), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[utils.matthews_correlation])
    # Train, train, train
    model.fit(train_X, train_y, batch_size=batchsize, epochs=epoch, validation_data=[val_X, val_y], callbacks=[ckpt])
    # loads the best weights saved by the checkpoint
    model.load_weights(os.path.join(result_dir, f'weights_{idx}.h5'))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict(val_X, batch_size=512))
    # and the val true y
    y_val.append(val_y)

# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[...,0]
#y_val = np.concatenate(y_val).flatten()
y_val = np.concatenate(y_val)

best_sarch_result = utils.threshold_search(y_val, preds_val)
best_threshold = best_sarch_result['threshold']
best_val_score = best_sarch_result['matthews_correlation']

print(f'best validation score: {best_val_score}')

testdata = VsbSignalDataset(mode='test')
X_test = feature_extracter.feature_extracter(feature_extracter.time_feature_extracter,
                                             testdata, window_size=window_size,
                                             stride=stride, grouped=True, 
                                             normalizer=normalizer)

print(X_test.shape)

np.save("X_test.npy",X_test)


submission = pd.read_csv('./data/sample_submission.csv')
print(len(submission))

preds_test = []
for i in range(N_SPLITS):
    model_path = os.path.join(result_dir, 'weights_{}.h5'.format(i))
    model.load_weights(model_path)
    pred = model.predict(X_test, batch_size=300, verbose=1)
    print(pred.shape)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(3):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)
print(np.array(preds_test).shape)

preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)

submission['target'] = preds_test
submission_filename = os.path.join(result_dir, f'submission_val{best_val_score:.4f}.csv')
submission.to_csv(submission_filename, index=False)
print(f'save result to {submission_filename}')



