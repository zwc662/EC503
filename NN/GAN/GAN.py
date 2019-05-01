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
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 400)
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

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.1, 0.5))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

dataset = data_proc.create_dataset('./train_ones.csv')
dataloader = DataLoader(dataset, batch_size = opt.batch_size, shuffle = True)
model_name = 'GAN'

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
    
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
    
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            loss_G = adversarial_loss(discriminator(gen_imgs), valid)

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_conf = discriminator(real_imgs)
            real_loss = adversarial_loss(real_conf, valid)
            epoch_real_conf += torch.mean(real_conf)

            fake_conf = discriminator(gen_imgs.detach())
            fake_loss = adversarial_loss(fake_conf, fake)
            epoch_fake_conf += torch.mean(fake_conf)

            loss_D = (real_loss + fake_loss) / 2

            loss_D.backward()
            optimizer_D.step()
    
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
    Y = np.ones([num_ones, 1])

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
    df.to_csv('./train_GAN.csv', sep = ',', header = None)
    return X, Y
        



if __name__ == "__main__":
    epoch = train()
    generate_ones('checkpoints/GAN_G.pt')
    
