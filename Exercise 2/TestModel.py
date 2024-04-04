# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:59:00 2024

@author: joche
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ModelClass import AutoencoderFNN, CAE
import torch.nn as nn
import pickle
from KClusterAlgoritm import getPlotPredictiveScore
from scipy.stats import multivariate_normal

from torchviz import make_dot


def GetData(CNN = False):
    # Open the file containing the saved data
    with open("stockTraindata.pkl", "rb") as f:
        # Load the data from the file
        trainData = pickle.load(f)
    with open("stockTestdata.pkl", "rb") as f:
        # Load the data from the file
        testData = pickle.load(f)
        
    if CNN:
        trainData = trainData.unsqueeze(1)
        testData = testData.unsqueeze(1)
    else:
        pass
    
    return trainData, testData
        
def testResultsModel(model, data, n = 0, CNN = True):
    """ 
    Return a plot of the models guess on input n.
    Set n to which data we check,
    TestOnTrain = bool
    CNN = bool, is it CNN data (extra channel channel)
    """
    outputs = model(data)
    
    print("Data is of shape:", data.squeeze().shape)
    
    # Need to squeeze the output to the right shape
    if CNN:
        outputs = outputs.squeeze()
        
    print("Output shape is : ", outputs[n].shape)
    
    print(data[n].shape)
    latentspace = model.encoder(data.unsqueeze(1)[n])
    
    print("latent space is of space: ", latentspace.shape)
    
    # Test specific case
    
    #Squeeze train and test data to view
    
    plt.figure()
    plt.title("Convolutional Autoencoder (CAE)")
    if CNN:
        plt.plot(data.squeeze().detach()[n], label = "Input")
    else:
        plt.plot(data.detach()[n], label = "Input")
    plt.plot(outputs.detach()[n], label = "Output")
    plt.legend()


def generate_pdf(points, bandwidth):
    # Fit a Gaussian kernel density estimator
    kde = multivariate_normal(mean=np.mean(points, axis=0), cov=bandwidth)
    return kde

def sample_points(pdf, num_samples):
    # Sample points from the PDF
    samples = pdf.rvs(size=num_samples)
    return samples



input_size = 150
hidden_size = 64
latent_space_size = 32


# Import the model
ModelName = 'model_V1.pth'
# model = AutoencoderFNN(input_size, hidden_size, latent_space_size)
model = CAE()
model.load_state_dict(torch.load(ModelName))
model.eval()

# Load in data, set CNN to true if we use CNN, this transformes the data to get an extra dim (channel dimension.)
trainData, testData = GetData(CNN = True)


print(trainData.shape)
# Get the output on train or test data
testResultsModel(model,trainData,n = 33)
testResultsModel(model,testData,n = 13)


# Try to generate a new sample

number_gen = 3
n = 21
latentspace = model.encoder(trainData[n])

#Create number_gen random vectors.
tensors = [torch.randn(latentspace.shape) for _ in range(number_gen)]
# Stack the tensors along a new dimension (dimension 0)
stacked_random_tensor = torch.stack(tensors, dim=0)

deviation = 1
# Either look around a solution or look at random generations.
latentspace = stacked_random_tensor
# latentspace = latentspace + deviation* stacked_random_tensor

Generateds = model.decoder(latentspace)


plt.figure()
for i in Generateds:
    plt.plot(i.squeeze().detach().numpy())
    
plt.title("randomly generated samples")
plt.show()




latentspace = model.encoder(trainData).view(384,-1)

print(latentspace.shape)



def getnewPointsBasedOnPDFofTrainData(N,latentspaceTrain):
    # Example usage
    # Define your points in N-dimensional space (350 points)
    N_dim = latentspaceTrain.shape[-1]
    print(N_dim)
    bandwidth_value = 0.05
    # Choose a bandwidth for the Gaussian kernel
    bandwidth = np.eye(N_dim) * bandwidth_value

    # Generate the PDF
    pdf = generate_pdf(latentspaceTrain.detach().numpy(), bandwidth)
    
    # Number of new points to sample
    num_samples = N

    # Sample new points from the PDF
    new_points = sample_points(pdf, num_samples)
    
    return new_points


N = 3
x = getnewPointsBasedOnPDFofTrainData(N,latentspace)


x = torch.tensor(x, dtype=torch.float32).view(N, 32,37)
Generateds = model.decoder(x)


plt.figure()
for i in Generateds:
    plt.plot(i.squeeze().detach().numpy())
    plt.title("randomly generated samples around mean of the distribution ")





# We do this because our model also works like this.
# x = (x) + latentspace

# print(x)

# Generateds = model.decoder(x)

# plt.figure()
# plt.title("Pertubations around testdata point")
# for gen in Generateds:
#     plt.plot(gen.detach())
    
    

# plt.plot(model.decoder(latentspace).detach())



"Test k clusters in Latenspace"

TrainDataInLatentSpace  = (model.encoder(trainData)).view(384,-1)

print(TrainDataInLatentSpace.shape)


ks = range(1,10,1)
# getPlotPredictiveScore(ks,TrainDataInLatentSpace.detach().numpy(), ratio_split=0.5, numberOfIteration=10)


from sklearn.manifold import TSNE


data = model.encoder(trainData).view(384,-1).detach().numpy()

print(np.shape(data))
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
data_2d_tsne = tsne.fit_transform(data)


# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], s=10)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('2D t-SNE Visualization of 32-dimensional Data')
plt.grid(True)
plt.show()

data = model.encoder(testData).detach().numpy()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
data_2d_tsne = tsne.fit_transform(data)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(data_2d_tsne[:, 0], data_2d_tsne[:, 1], s=10)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('2D t-SNE Visualization of 32-dimensional Data')
plt.grid(True)
plt.show()