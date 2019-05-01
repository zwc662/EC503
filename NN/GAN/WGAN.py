import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import pandas as pd
import data_proc


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc3 = nn.Linear(400, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x



class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Initialize generator and discriminator
generator = Generator(opt.latent_dim, 200)
discriminator = Discriminator(200, 1)

if cuda:
    generator.cuda()
    discriminator.cuda()


# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

dataset = data_proc.create_dataset('./train_ones.csv')
dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = True)
model_name = 'WGAN'

def train(checkpoint = None):
    batches_done = 0

    i_epoch = 0
    if checkpoint is not None:
        checkpoint_G = torch.load(checkpoint[0])
        checkpoint_D = torch.load(checkpoint[1])

        generator.load_state_dict(checkpoint_G['model_state_dict'])
        discriminator.load_state_dict(checkpoint_D['model_state_dict'])

        optimizer_G.load_state_dict(checkpoint_G['optimizer_state_dict'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])

        if checkpoint_G['epoch'] > checkpoint_D['epoch']:
            i_epoch = checkpoint_G['epoch']
        else:
            i_epoch = checkpoint_D['epoch']

        loss_G = checkpoint_G['loss']
        loss_D = checkpoitn_D['loss']

        generator.eval()
        discriminator.eval()

        loss_G.backward()
        optimizer_G.step()

        loss_D.backward()
        optimizer_D.step()

    for epoch in range(opt.n_epochs):
    
        epoch_real_conf = 0.0
        epoch_fake_conf = 0.0

        for i, (imgs, _) in enumerate(dataloader):
    
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
            
            real_conf = torch.mean(discriminator(real_imgs).detach()) 
            fake_conf = torch.mean(discriminator(fake_imgs.detach()).detach())
            epoch_real_conf += real_conf
            epoch_fake_conf += fake_conf
    
            loss_D.backward()
            optimizer_D.step()
    
            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
    
            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
    
                # -----------------
                #  Train Generator
                # -----------------
    
                optimizer_G.zero_grad()
    
                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))
    
                loss_G.backward()
                optimizer_G.step()
                

    
    
            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Real conf: %f] [Fake conf: %f]"
                    % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item(), epoch_real_conf/(i + 1), epoch_fake_conf/(i + 1))
                )
            batches_done += 1

        print("[Epoch %d/%d] [Real conf: %f] [Fake conf: %f]" % (epoch, opt.n_epochs, epoch_real_conf/len(dataloader), epoch_fake_conf/len(dataloader)))
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': generator.state_dict(), 'optimizer_state_dict': optimizer_G.state_dict(), 'loss': loss_G}, str('checkpoints/' + model_name + '_G_' +  str(epoch) + '.pt'))
            torch.save({'epoch': epoch, 'model_state_dict': discriminator.state_dict(), 'optimizer_state_dict': optimizer_D.state_dict(), 'loss': loss_D}, str('checkpoints/' + model_name + '_D_' +  str(epoch) + '.pt'))
        
        if epoch > 50 and abs(epoch_real_conf/(i + 1) - 0.5) < 0.01 and abs(epoch_fake_conf/(i+1) - 0.5) < 0.01:
            torch.save({'epoch': epoch, 'model_state_dict': generator.state_dict(), 'optimizer_state_dict': optimizer_G.state_dict(), 'loss': loss_G}, str('checkpoints/' + model_name + '_G_' +  str(epoch) + '.pt'))
            torch.save({'epoch': epoch, 'model_state_dict': discriminator.state_dict(), 'optimizer_state_dict': optimizer_D.state_dict(), 'loss': loss_D}, str('checkpoints/' + model_name + '_D_' +  str(epoch) + '.pt'))
         
        if abs(epoch_real_conf/(i + 1) - 0.5) < 0.001 and abs(epoch_fake_conf/(i+1) - 0.5) < 0.001:
            torch.save({'epoch': epoch, 'model_state_dict': generator.state_dict(), 'optimizer_state_dict': optimizer_G.state_dict(), 'loss': loss_G}, str('checkpoints/' + model_name + '_G.pt'))
            torch.save({'epoch': epoch, 'model_state_dict': discriminator.state_dict(), 'optimizer_state_dict': optimizer_D.state_dict(), 'loss': loss_D}, str('checkpoints/' + model_name + '_D.pt'))

def generate_ones(checkpoint, num_ones = 150000):
    checkpoint = torch.load(checkpoint)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    
    X = np.zeros([num_ones, 200])
    Y = np.ones([num_ones,])

    for i in range(num_ones):
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        fake_imgs = generator(z).detach().cpu().numpy()
        X[i] = fake_imgs

    data = pd.read_csv('../train.csv', sep = ',', header = None)
    X_ = data.values[1:, 2:]
    Y_ = data.values[1:, 1]

    X = np.concatenate((X, X_), axis = 0)
    Y = np.concatenate((Y, Y_), axis = 0)
    print(X.shape)
    print(Y.shape)

    df = pd.DataFrame(np.concatenate((np.reshape(Y, [Y.shape[0], 1]), X), axis = 1))
    df.to_csv('./train_WGAN.csv', sep = ',', header = None)
    return X, Y
        



if __name__ == "__main__":
    #epoch = train()
    generate_ones('checkpoints/WGAN_G.pt')
    
