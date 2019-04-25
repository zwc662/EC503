import os
import numpy as np
import scipy.io as sio
import torch.utils.data as data_utils
import torch
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN

def preprocess_file(path = None):
    if path is None:
        path = '/home/depend/workspace/EC503/train.csv'
    data = pd.read_csv(path, sep = ',', header = None)
    X = data.values[1:, 2:]
    Y = data.values[1:, 1]

    size = X.shape
    print(size)
    X = np.asarray(X).astype(float)
    Y = np.reshape(Y, [size[0]]).astype(int)
    return X, Y

def preproc_data(path):
    X, Y = preprocess_file(path)
    X_mean = np.mean(X, axis = 0)
    X_std = np.std(X, axis = 0)
    X = (X - X_mean)/X_std
    df = pd.DataFrame(np.concatenate((np.reshape(Y, [Y.shape[0], 1]), X), axis = 1))
    df.to_csv('./train.csv', sep = ',', header = None)
    return X, Y

def create_dataset(path = None, proc_data = False, imb = None):
    if proc_data:
        X, Y = preproc_data(path)
    else:
        X, Y = preprocess_file(path)
    print(np.sum(Y, axis = 0))
    if imb is None:
        pass
    elif imb == 'SMOTE':
        X, Y = SMOTE().fit_resample(X, Y)
        df = pd.DataFrame(np.concatenate((np.reshape(Y, [Y.shape[0], 1]), X), axis = 1))
        df.to_csv('./train_SMOTE.csv', sep = ',', header = None)
    elif imb == 'ADASYN':
        X, Y = ADASYN().fit_resample(X, Y)
        df = pd.DataFrame(np.concatenate((np.reshape(Y, [Y.shape[0], 1]), X), axis = 1))
        df.to_csv('./train_ADASYN.csv', sep = ',', header = None)
 
    print(X.shape, Y.shape)
    print(np.sum(Y, axis = 0))


    return data_utils.TensorDataset(torch.from_numpy(np.asarray(X)), torch.from_numpy(np.asarray(Y)))




if __name__ == "__main__":
    create_dataset(proc_data = True, imb = 'SMOTE')
    create_dataset(proc_data = True, imb = 'ADASYN')

