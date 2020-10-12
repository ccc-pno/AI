import numpy as np

def rbf_kernel(X,y,gamma=None):
    """对点集X计算两两之间的高斯值"""
    if gamma==None:
        gamma = 1.0/X.shape[1]
    affinity_matrix = np.zeros((len(X),len(y)))
    for i in range(len(X)):
        for j in range(len(y)):
            diff = np.sum((X[i,:]-y[j,:])**2)
            affinity_matrix[i][j] = np.exp(-gamma*diff)
    return affinity_matrix