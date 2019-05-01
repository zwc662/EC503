import numpy as np
import scipy.io as sio
import torch.utils.data as data_utils
import torch
import pandas as pd

def data_process(path = "../train.csv"):
    data = pd.read_csv(path, sep = ',', header = None)
    X = data.values[1:, 2:]
    Y = data.values[1:, 1]

    size = X.shape
    print(size)
    X = np.asarray(X).astype(float)
    Y = np.reshape(Y, [size[0]]).astype(int)
    num_ones = np.sum(Y, axis = 0)
    X_ = np.zeros(X.shape)
    Y_ = np.ones(Y.shape)
    for i in range(Y.shape[0]):
        if Y[i] == 1:
            X_[i] = X[i]
     
    df = pd.DataFrame(np.concatenate((np.reshape(Y_, [Y_.shape[0], 1]), X_), axis = 1))
    df.to_csv('./train_ones.csv', sep = ',', header = None)

def create_dataset(path = './train_ones.csv'):
    data = pd.read_csv(path, sep = ',', header = None)
    X = data.values[1:, 2:]
    Y = data.values[1:, 1]
    return data_utils.TensorDataset(torch.from_numpy(np.asarray(X)), torch.from_numpy(np.asarray(Y)))
    
def generate_dataset():
    data = pd.read_csv('../train.csv', sep = ',', header = None)
    X = data.values[1:, 2:]
    Y = data.values[1:, 1]

    data = pd.read_csv('./train_GAN_ones.csv', sep = ',', header = None)
    X_ = data.values[1:, 2:]
    Y_ = data.values[1:, 1]

    X = np.concatenate((X, X_), axis = 0)
    Y = np.concatenate((Y, Y_), axis = 0)
    print(X.shape)
    print(Y.shape)

    df = pd.DataFrame(np.concatenate((np.reshape(Y, [Y.shape[0], 1]), X), axis = 1))
    df.to_csv('./train_GAN.csv', sep = ',', header = None)
    return X, Y

