import pyarrow.parquet as pq
import os 
import numpy as np
import random as rn
# fix random seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
import tensorflow as tf
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                              gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
from keras import backend as K
K.set_session(sess)
import torch
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.deterministic = True


import argparse
from datetime import datetime
import pandas as pd
from keras.layers import *
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from keras import optimizers
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from keras.callbacks import *

import utils
from vsb_signal_dataset import VsbSignalDataset
from normalizer import Normalizer
import feature_extracter
from models import modelutils


parser = argparse.ArgumentParser(description='vsb-power-line-fault-detection on kaggle')
parser.add_argument('--config', '-c', type=str, default='./config/base_config.json',
                    help='path to config file')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--no_cache", help="extract feature without cache",
                    action="store_true")
args = parser.parse_args()

if args.debug:
    result_dir = os.path.join(utils.RESULT_DIR, 'debug-'+datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
else:
    result_dir = os.path.join(utils.RESULT_DIR, datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
os.mkdir(result_dir)
print(f'created: {result_dir}')

# load config
config = utils.load_config(args.config)

N_SPLITS = config['N_SPLITS']
epoch = config['epoch']
if args.debug:
    print('running debug mode')
    epoch = 5
    N_SPLITS = 2
    
batchsize = config['batch_size']
sample_size = 800000
max_num = config['max_num']
min_num = config['min_num']
#n_dim = 160
window_size = config['window_size']
stride = config['stride']
grouped = config['grouped']
model_name = config['model_name']
features = config['features']

lr = config['lr']
optimizer = config['optimizer']
loss = config['loss']

save_feature = False if args.debug else True
use_cache = False if args.no_cache else True

# dump config
utils.save_config(config, os.path.join(result_dir, 'config.json'))

dataset = VsbSignalDataset(mode='train', debug=args.debug)
normalizer = Normalizer(min_num, max_num)

X = feature_extracter.feature_extracter(features,
                                        dataset, window_size=window_size, stride=stride,
                                        grouped=grouped, normalizer=normalizer, use_cache=use_cache,
                                        save_result=save_feature)

y = dataset.labels
#y = y[::3].reshape(-1, 1)
y = y[::3]

print(X.shape, y.shape)
np.save(os.path.join(result_dir, "X.npy"),X)
np.save(os.path.join(result_dir, "y.npy"),X)


#splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019).split(X, y))
if args.debug:
    splits = dataset.get_folds(n_splits=N_SPLITS)
else:
    adversarial_train_group = np.load('data/adversarial_train_group.npy')
    adversarial_val_group = np.load('data/adversarial_val_group.npy')
    splits = [(adversarial_train_group, adversarial_val_group)] * N_SPLITS

preds_val = []
y_val = []

for idx, (train_idx, val_idx) in enumerate(splits):
    print("Beginning fold {}".format(idx+1))
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    model = modelutils.get_model(model_name, train_X.shape)
    ckpt = ModelCheckpoint(os.path.join(result_dir, f'weights_{idx}.h5'), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                   verbose=1, mode='min', epsilon=0.0001)
    early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=25)
    model.compile(loss=loss, optimizer=optimizer, metrics=[utils.matthews_correlation])
    K.set_value(model.optimizer.lr, lr)
    model.fit(train_X, train_y, batch_size=batchsize, epochs=epoch, validation_data=[val_X, val_y], callbacks=[ckpt, reduce_lr, early])
    model.load_weights(os.path.join(result_dir, f'weights_{idx}.h5'))
    preds_val.append(model.predict(val_X, batch_size=512))
    y_val.append(val_y)

# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[...,0]
#y_val = np.concatenate(y_val).flatten()
y_val = np.concatenate(y_val)
print(preds_val.shape)
print(y_val.shape)

best_sarch_result = utils.threshold_search(y_val, preds_val)
best_threshold = best_sarch_result['threshold']
best_val_score = best_sarch_result['matthews_correlation']

print(f'best threshold: {best_threshold}')
print(f'best validation score: {best_val_score}')

testdata = VsbSignalDataset(mode='test')
X_test = feature_extracter.feature_extracter(features,
                                             testdata, window_size=window_size,
                                             stride=stride, grouped=grouped, 
                                             normalizer=normalizer, use_cache=use_cache,
                                             save_result=save_feature)

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






