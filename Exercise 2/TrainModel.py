# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:15:06 2024

@author: joche
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ModelClass import AutoencoderFNN, CAE
import pickle
np.random.seed(40)

def GetData(ratiosplit = 0.8):
    "Function to load in an split the data into train an test data"
    data = np.load("stockData.npy", allow_pickle=True)

    # Shuffle the data
    np.random.shuffle(data)

    # Prepare data
    lengthData = len(data)
    ratioTrain =  int(lengthData * ratiosplit)
    # Split data
    full_train_data = torch.Tensor(data[0:ratioTrain])
    full_test_data = torch.Tensor(data[ratioTrain:])

    # Save this to a pickle file for testing purposes
    with open("stockTraindata.pkl", "wb") as f:
        pickle.dump(full_train_data,f)
    with open("stockTestdata.pkl", "wb") as f:
        pickle.dump(full_test_data,f)
        
    return full_train_data, full_test_data



# Define input and hyperparameters
input_size = 150
hidden_size = 64
latent_space = 32

# Create model instance
# model = AutoencoderFNN(input_size, hidden_size, latent_space)

# The parameters are in the file ModelClass
model = CAE()

# Print model summary
print(model)

# Intitilise data
full_train_data, full_test_data = GetData()
# Get data in right format.
train_data_loader = DataLoader(full_train_data, batch_size=32,  shuffle=True)
test_data_loader = DataLoader(full_test_data,batch_size= 32, shuffle = True)

plt.figure()
plt.title("Traindata")
for i in full_train_data:
    plt.plot(i)



"Train procedure"
learning_rate = 10 ** -4
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# Save the losses to plot later
train_losss = []
test_losss  = []

# Main train loop
num_epoch = 1000
for epoch in range(num_epoch):
    running_trainloss = 0.0
    running_testloss = 0.0
    for data_train, data_test in zip(train_data_loader,test_data_loader):
        
        "Remove this if FNN is used, only nesceary for CNN."
        # Need to add an extra channel dimension, use unsqueeze to add dimension at between dim 0 , and dim 1.
        data_train = data_train.unsqueeze(1)
        data_test = data_test.unsqueeze(1)
        
        # Set gradient of all parameters to zero
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data_train)
        
        # print(outputs.shape)
        
        alpha = 3

        # Compute loss, using that the labels are our inputs, autoencoder
        loss = criterion(outputs, data_train) + alpha * torch.sum(outputs[:,:,0] - 1)**2
        running_trainloss += loss.item()
        # Calculate the gradient with respect to the loss
        loss.backward()
        # optimise the parameters
        optimizer.step()
        
        # Test on test data
        outputs = model(data_test)
        loss = criterion(outputs,data_test) + alpha * torch.sum(outputs[:,:,0] - 1)**2
        
        running_testloss += loss.item()
        
        
    print(f'Epoch {epoch+1}, TrainLoss: {running_trainloss / len(data_train)}')
    print(f'Epoch {epoch+1}, TestLoss: {running_testloss / len(data_test)}')
    print("/n")
    
    train_losss.append(running_trainloss)
    test_losss.append(running_testloss)


plt.figure()
plt.title("Loss training")
plt.plot(train_losss, label = "Training")
plt.plot(test_losss, label = "Test")
plt.legend()


# Save model
torch.save(model.state_dict(), 'model_V1.pth')

    
    