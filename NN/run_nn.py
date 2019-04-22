import argparse
from pathlib import PurePath as Path
import scipy.optimize

from data_proc import *
from model import mlp

import torch
from torch.autograd import Variable
import torch.utils.data as data_utils

import logging
import time
import os

import pickle


size_input = 200
size_output = 2
size_batch = 25

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = create_dataset(path = 'train_SMOTE.csv')

net = mlp(size_input, size_output).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum = 0.9)


for epoch in range(50):  # loop over the dataset multiple times
    dataloader = data_utils.DataLoader(dataset, batch_size = size_batch, shuffle = True)

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = torch.reshape(outputs, (outputs.size()[0], size_output))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

tp = 0
fp = 0
tn = 0
fn = 0

test = True
dataset = create_dataset()
dataloader = data_utils.DataLoader(dataset, batch_size = size_batch, shuffle = True)
corr = 0
tot = 0
if test == True:
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.float().to(device)

        outputs = net(inputs).detach()
        outputs = torch.squeeze(torch.argmax(torch.reshape(outputs, (size_batch, size_output)), 1)).cpu().numpy()
        labels = labels.numpy()

        corr = corr + np.sum((outputs == labels).astype(int))
        tot = tot + outputs.shape[0]
       
        tp += np.sum((outputs == 1) * (labels == 1))
        fp += np.sum((outputs == 1) * (labels == 0))
        tn += np.sum((outputs == 0) * (labels == 0))
        fn += np.sum((outputs == 0) * (labels == 1))

print("Accuracy: %f" % corr * 1.0/tot)
print(tp, fp, tn, fn)
print("Precision: {}\n Recall: {}\n".format(tp * 1.0/(tp + fp * 1.0), tp * 1.0/(tp + fn * 1.0)))
