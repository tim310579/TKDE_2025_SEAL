import os
import sys
import re
import glob
import pickle
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
import ast
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import warnings

def load_dataset(path, sampling_rate, label_num, release=False):
    print(path)
    
    if path.split('/')[-2] == 'CPSC_extra':
        Y = pd.read_csv(path+'CPSC_extra_bk_stat.csv', index_col='id')

        X = load_raw_data_CPSC_extra(Y, sampling_rate, path)

    elif path.split('/')[-2] == 'Chapman':
        Y = pd.read_csv(path+'Chapman_bk_stat.csv', index_col='id')

        X = load_raw_data_Chapman(Y, sampling_rate, path)

    return X, Y

def load_raw_data_Chapman(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path +str(f)) for f in tqdm(df.filename)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data


def load_raw_data_CPSC_extra(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path +str(f)) for f in tqdm(df.filename)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def select_data(XX,YY, ctype, class_name, label_num, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'all':
        
        if class_name == 'CPSC_extra':
            counts = YY['label%d'%label_num].value_counts()
            counts = counts[counts > min_samples]
            YY['label_len'] = YY['label%d'%label_num].apply(lambda x: len(x))
            X = XX[YY.label_len > 0]
            Y = YY[YY.label_len > 0]
            mlb.fit(Y['label%d'%label_num].values)
            y = mlb.transform(Y['label%d'%label_num].values)

        elif class_name == 'Chapman':
            counts = YY['label%d'%label_num].value_counts()
            counts = counts[counts > min_samples]
            YY['label_len'] = YY['label%d'%label_num].apply(lambda x: len(x))
            X = XX[YY.label_len > 0]
            Y = YY[YY.label_len > 0]
            mlb.fit(Y['label%d'%label_num].values)
            y = mlb.transform(Y['label%d'%label_num].values)

    else:
        pass

    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb

