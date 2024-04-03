# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:03:35 2024

@author: joche
"""

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from Standard_GAN import *
import matplotlib.pyplot as plt




def train_gan(G, D, data_loader, gan_type='vanilla', num_epochs=100, batch_size=64, learning_rate=0.0001, ratio_train_D_G=1, input_dim_G=100):
    # Determine the loss function and optimizer based on GAN type
    if gan_type == 'vanilla':
        criterion = nn.BCELoss()
    elif gan_type == 'wasserstein':
        criterion = lambda output, target: -torch.mean(output) if target == 1 else torch.mean(output)
    
    else:
        raise ValueError("Unsupported GAN type. Choose 'vanilla' or 'wasserstein'.")

    # Set up optimisers
    optimizer_G = Adam(G.parameters(), lr=learning_rate)
    print(D)
    optimizer_D = Adam(D.parameters(), lr=learning_rate)
    
    lossD_array = []
    lossG_array = []

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            real_data = data.float()
            batch_size = real_data.size(0)
            
            # Labels for real and fake data
            real_labels = torch.full((batch_size, 1), 1, dtype=torch.float) if gan_type == 'vanilla' else None
            fake_labels = torch.full((batch_size, 1), 0, dtype=torch.float) if gan_type == 'vanilla' else None

            # Train Discriminator/Critic
            optimizer_D.zero_grad()

            # Real data loss
            output_real = D(real_data)
            loss_real = criterion(output_real, real_labels) if gan_type == 'vanilla' else criterion(output_real, 1)

            # Fake data
            z = torch.randn(batch_size, input_dim_G)
            fake_data = G(z)
            output_fake = D(fake_data.detach())
            loss_fake = criterion(output_fake, fake_labels) if gan_type == 'vanilla' else criterion(output_fake, 0)

            # Total Discriminator/Critic loss
            loss_D = (loss_real + loss_fake) / 2 if gan_type == 'vanilla' else loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            if gan_type == 'wasserstein':
                for p in D.parameters():
                    p.data.clamp_(-0.1, 0.1)

            lossD_array.append(loss_D.item())

            # Train Generator
            if i % ratio_train_D_G == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, input_dim_G)
                fake_data = G(z)
                output = D(fake_data)
                loss_G = criterion(output, real_labels) if gan_type == 'vanilla' else criterion(output, 1)
                loss_G.backward()
                optimizer_G.step()
                
                lossG_array.append(loss_G.item())

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], "
                      f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    generator_path = 'generator1.pth'
    discriminator_path = 'discriminator1.pth'

    # Save Generator model
    torch.save(G.state_dict(), generator_path)

    # Save Discriminator model
    torch.save(D.state_dict(), discriminator_path)

    return lossD_array, lossG_array


# Load in the data
traindata =  np.load("TrainDataExcersise1.npy", allow_pickle = True)
print(np.shape(traindata))
# Convert to a Tensor
traindata = torch.tensor(traindata)

# Hyperparameters
# Input Output generator
input_dim_G = 2  # Latent space dimension
output_dim_G = 2 # Outputgenerator

# Input Output discriminator
input_dim_D  = output_dim_G 
output_dim_D = 1

# Models
G = Generator(input_dim_G, output_dim_G)
# D = Discriminator(input_dim_D, output_dim_D)

C = Critic(input_dim_D)


# If we wish to continue training, reload the models
generator_path = 'generator1.pth'
discriminator_path = 'discriminator1.pth'
G.load_state_dict(torch.load(generator_path))
# D.load_state_dict(torch.load(discriminator_path))
C.load_state_dict(torch.load(discriminator_path))
# parameters for learning process
learning_rate = 0.0001
batch_size = 64
num_epochs = 100
ratio_train_D_G = 30

# Optimizers
optimizer_G = Adam(G.parameters(), lr=learning_rate)
# optimizer_D = Adam(D.parameters(), lr=learning_rate)

# Loss function Binary cross entropy.
criterion = nn.BCELoss()


# Data loader
data_loader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle=True, drop_last=True)
# Training loop

# Traini ng
lossD_array, lossG_array = train_gan(G, C, data_loader, gan_type='wasserstein', num_epochs=num_epochs, batch_size=batch_size, learning_rate=0.0001, ratio_train_D_G=ratio_train_D_G, input_dim_G=input_dim_G)


# Assuming lossD_array and lossG_array are lists of PyTorch tensors
lossD_numpy_list = [tensor for tensor in lossD_array]
lossG_numpy_list = [[tensor for _ in range(ratio_train_D_G-4)] for tensor in lossG_array]


lossG_numpy_list = np.concatenate(lossG_numpy_list)


plt.figure()
plt.plot(lossD_numpy_list, label='Loss Critic')
plt.plot(lossG_numpy_list, label='Loss Generator')
plt.title('Training Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
