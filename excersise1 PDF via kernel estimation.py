# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:09:20 2024

@author: jochem
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def estimationPDF(x, data, N, b = 1):
    "Gives the estimated pdf for x given train data, and a hyper parameter b"
    "Also give the length of the dataset N, do not wish to compute this every time."
    z  = np.sum(((x - data)/b) **2,axis = 1)
    kernels = kernel(z)
    return np.sum(kernels) /(N*b)
    

def kernel(z):
    "return a similarity for two points z = x1 -x2/b"
    return (1 / np.sqrt((2* np.pi)))  * np.exp(-z / 2)

# Correcting the code to initialize 'data' as a 2-dimensional array

n = 10000
# Generate 6 means points on the unit circle.
gaussian_means = [[np.cos(2*np.pi*i/6), np.sin(2*np.pi*i/6)] for i in range(6) ]

# Initialize 'data' as an empty array
data = np.empty((0, 2))
# Change the variance or sigma to get wider circles
var = 0.05
for i in range(n):
    # Get random index
    idx = np.random.randint(0, 6)
    # Create a sample form the normal distibution with variance 1 but because we multiply with 0.05 we get 
    new_data = var*np.random.randn(1, 2) + gaussian_means[idx]
    data = np.concatenate((data, new_data), axis=0)



np.save("TrainDataExcersise1.npy", data)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))


Z_custom = np.zeros_like(X)

b = 0.1
# Loop through each point in the grid and apply your custom function
for i in range(len(x)):
    for j in range(len(y)):
        Z_custom[i, j] = estimationPDF(pos[i,j], data, n, b = b)#pos[i, j], data, n, b = 0.1)


plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

plt.contourf(X, Y, Z_custom, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('b = ' + str(b) + ", N = " + str(n), size = 15)



# plt.show()
plt.show()