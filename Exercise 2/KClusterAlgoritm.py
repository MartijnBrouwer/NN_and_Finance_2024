# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:39:46 2024

@author: joche
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

import os

# Set OMP_NUM_THREADS environment variable to 1
os.environ["OMP_NUM_THREADS"] = "1"


def predictiveScore(k, predict_X_2_with_train, predict_X_2_with_test):
    "Returns a predictive score k number of cluster"
    
    # Define K lists where we save the indicies of the datapoints belonging to cluster K.
    IndiciesMatrix = [[] for _ in range(k)]
    
    # O(N)
    for index,cluster in enumerate(predict_X_2_with_test):
        IndiciesMatrix[cluster].append(index)
        
    scoresForEachClass = []
    for i in range(k):
        # Length of the cluster j of the test set
        lenAj  = len(IndiciesMatrix[i])
        
        # Get the classes predicted by the trainset of the testset
        lst = (predict_X_2_with_train[IndiciesMatrix[i]])
        # Count how often each classes is present
        frequency = Counter(lst)

        if lenAj - 1 == 0:
            print("A class had only one child, thus we skipped this one")
        else:
            # Calculate the number of pairs possible 
            agrementBetweentrainandtest = sum(count * (count - 1) for count in frequency.values()) / (lenAj * lenAj -1)
            scoresForEachClass.append(agrementBetweentrainandtest)
        
    return min(scoresForEachClass)
        
def PredictiveScoreLoop(ks, X_1, X_2): 
    "A function that return the predictive score for data an a range of ks"
    predictiveStrength = []
    for k in ks:
        kmeans1 = KMeans(n_clusters=k)
        kmeans1.fit(X_1)
        
        predict_X_2_with_train = kmeans1.predict(X_2)
            
        kmeans2 = KMeans(n_clusters=k)
        kmeans2.fit(X_2)
        predict_X_2_with_test = kmeans2.predict(X_2)
        
        
        
        score = predictiveScore(k,predict_X_2_with_train, predict_X_2_with_test)
        predictiveStrength.append(score)
        
    return predictiveStrength
    
def getPlotPredictiveScore(ks, data, ratio_split = 0.5, numberOfIteration = 5):
    """ 
    A function that return a plot where we can see what the best k is
    
    Parameters: 
            ks:  is the the range of ks we would like to investigate
            data: high dim 2d data in numpy array (number of sample, featers)
            ratio_split: the split of our data in train and test, i believe 0.5 is best
            n: how often do we run the k clustering to get stable results, k clustering depends a lot on initial chose of the mu's so we try to minimise this impact'
    """
    
    data = data
    np.random.shuffle(data)
    
    n = len(data)
    
    # Divide the data up into sets of random data.
    X_1 = data[0:int(n*ratio_split)]
    X_2 = data[int(n*ratio_split):]
    
    # Check 1
    if len(X_1) ==0  or len(X_2) == 0:
        print("To little data provided or the split ratio is to high.")
    

    full_predictiveStrength = []
    # We loop over multiple simulation to get a good estimate for the predictive strength
    # Note that K clustering is a random process that is sensible to begin conditions.
    for _ in range(numberOfIteration):
        scores = PredictiveScoreLoop(ks,X_1,X_2)
        
        full_predictiveStrength.append(scores)
        
        
    final_predictiveStrength = (np.mean(full_predictiveStrength,axis = 0))
        
    
    
    plt.figure()
    plt.title("K Clustering predictive score")
    plt.plot(ks,final_predictiveStrength)
    plt.xlabel("k's")
    plt.ylabel("Predictive score")
    
