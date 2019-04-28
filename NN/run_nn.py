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

net = mlp(size_input, size_output).to(device)

def train(path = 'train_SMOTE.csv', num_epoch = 100):
    dataset = create_dataset(path)
    
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum = 0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    
    for epoch in range(num_epoch):  # loop over the dataset multiple times
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
        
        torch.save(net.state_dict(), str('checkpoints/nn' + str(epoch) + '.pt'))
    
    print('Finished Training')


def test(path = 'train.csv', checkpoint = 'checkpoints/nn1.pt'):
    
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint)

    net.eval()

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    test = True
    dataset = create_dataset(path)
    dataloader = data_utils.DataLoader(dataset, batch_size = size_batch, shuffle = True)
    corr = 0
    tot = 0
    if test == True:
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.float().to(device)
    
            outputs = net(inputs).detach()
            outputs = torch.squeeze(torch.argmax(outputs, 1)).cpu().numpy()
            labels = labels.numpy()
    
            corr = corr + np.sum((outputs == labels).astype(int))
            tot = tot + outputs.shape[0]
           
            tp += np.sum((outputs == 1) * (labels == 1))
            fp += np.sum((outputs == 1) * (labels == 0))
            tn += np.sum((outputs == 0) * (labels == 0))
            fn += np.sum((outputs == 0) * (labels == 1))
    
    print("Accuracy: %f" % (corr * 1.0/tot))
    print("True Positive Number: {}\nFalse Positive Number: {}\nTrue Negative Number: {}\nFalse Negative Number: {}".format(tp, fp, tn, fn))
    print("Precision: {}\nRecall: {}\n".format((tp * 1.0/(tp + fp * 1.0)), (tp * 1.0/(tp + fn * 1.0))))


if __name__ == '__main__':
    #train()
    test(checkpoint = 'checkpoints/nn99.pt')
