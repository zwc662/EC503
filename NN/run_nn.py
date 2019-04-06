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

dataset = create_dataset()
dataloader = data_utils.DataLoader(dataset, batch_size = size_batch, shuffle = True)

net = mlp(size_input, size_output).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum = 0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        outputs = torch.reshape(outputs, (size_batch, size_output))

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


pos_lab = 0
pos_pred = 0
neg_lab = 0
neg_pred = 0

if True:
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data

        outputs = net(inputs.float()).detach()
        outputs = torch.squeeze(torch.argmax(torch.reshape(outputs, (size_batch, size_output)), 1)).numpy()
        print(outputs)
        labels = labels.numpy()


        pos_lab = pos_lab + np.sum(labels == 1)
        pos_pred = pos_pred + np.sum(outputs == 1)
        neg_lab = neg_lab + np.sum(labels == 0)
        neg_pred = neg_pred + np.sum(outputs == 0)

print(pos_lab, pos_pred, neg_lab, neg_pred)
