# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:36:32 2024

@author: jochem
"""

""" 
GAN Frame work
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__() # Initilise the parent Generator class
        self.net = nn.Sequential(
            # Adjust this architecture
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__() # Initilise the parent Discriminator class
        self.net = nn.Sequential(
            # Adjust this architecture
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)




class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            # No Sigmoid activation at the end
        )

    def forward(self, x):
        return self.net(x)


