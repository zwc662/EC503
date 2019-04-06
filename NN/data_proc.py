import os
import numpy as np
import scipy.io as sio
import torch.utils.data as data_utils
import torch
import pandas as pd


def preprocess_file():
    data = pd.read_csv("train.csv", sep = ',', header = None)
    X = data.values[1:, 2:]
    Y = data.values[1:, 1]

    size = X.shape
    print(size)
    X = np.asarray(X).astype(float)
    Y = np.reshape(Y, [size[0]]).astype(int)
    return X, Y

def preproc_data():
    X, Y = preprocess_file()
    X_max = np.max(X, axis = 0)
    X = X/X_max
    return X, Y

def create_dataset():
    X, Y = preproc_data()
    return data_utils.TensorDataset(torch.from_numpy(np.asarray(X)), torch.from_numpy(np.asarray(Y)))

if __name__ == "__main__":
    create_dataset()
