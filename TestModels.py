# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:55:27 2024

@author: joche
"""

import torch
from Standard_GAN import Generator, Discriminator, Critic
import numpy as np
import matplotlib.pyplot as plt


# Define paths to save the models
generator_path = 'generator1.pth'
discriminator_path = 'discriminator1.pth'

# Hyperparameters
# Input Output generator
input_dim_G = 2# Latent space dimension
output_dim_G = 2 # Outputgenerator

# Input Output discriminator
input_dim_D  = output_dim_G # Latent space dimension
output_dim_D = 1


# Save Generator model
# Load Generator model architecture and weights from saved file
G = Generator(input_dim_G, output_dim_G)  # Initialize with dummy dimensions
G.load_state_dict(torch.load(generator_path))

# Extract input and output dimensions from the loaded Generator model
input_dim_G_loaded = G.net[0].in_features
output_dim_G_loaded = G.net[-1].out_features


# Load Discriminator model architecture and weights from saved file
D = Discriminator(input_dim_D,output_dim_D)
D.load_state_dict(torch.load(discriminator_path))

C = Critic(input_dim_D)
C.load_state_dict(torch.load(discriminator_path))

# Uncomment if Wasserstein
D = C

batch_size = 300

z = torch.randn(batch_size, input_dim_G)
fake_data = G(z).detach()

guess_fake = D(fake_data).detach()


realData =  np.load("TrainDataExcersise1.npy", allow_pickle = True)[0:300]


x = np.linspace(-2, 2, num=100)  # Adjust the number of points as needed
y = np.linspace(-2, 2, num=100)

# Create the grid
X, Y = np.meshgrid(x, y)

# Combine X and Y into a single array
grid_points = np.column_stack((X.flatten(), Y.flatten()))

print(torch.tensor(realData, dtype=torch.float32))
guess_real = D(torch.tensor(realData,  dtype=torch.float32)).detach()

guess_grid = D(torch.tensor(grid_points, dtype= torch.float32)).detach()


# Plot distribution discriminator
plt.figure()
plt.scatter(grid_points[:, 0], grid_points[:,1],c = guess_grid, cmap= 'RdYlGn')
# plt.title('Discriminator probablility real vs fake')
plt.title('Critic score')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Values (0 to 1)')
plt.grid()

plt.figure()
plt.scatter(realData[:, 0], realData[:,1],c = guess_real, cmap= 'RdYlGn', label = "real data")
# plt.scatter(fake_data[:, 0], fake_data[:, 1], c=guess_fake, cmap='RdYlGn',label = "Fake Data")
plt.title('6 2D gaussians lying on a circle')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Values (0 to 1)')
plt.legend()
plt.grid()

plt.figure()
# plt.scatter(realData[:, 0], realData[:,1],c = guess_real, cmap= 'RdYlGn', label = "real data")
plt.scatter(fake_data[:, 0], fake_data[:, 1], c=guess_fake, cmap='RdYlGn')
plt.title('Generated data')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Values (0 to 1)')
plt.grid()


plt.figure()
plt.scatter(realData[:, 0], realData[:,1], label = "real data")
plt.scatter(fake_data[:, 0], fake_data[:, 1],label = "Fake Data")
plt.title('6 2D gaussians lying on a circle')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()






