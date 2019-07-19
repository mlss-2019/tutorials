import numpy as np
from sqDist import *


def kernelMatrix(X1, X2, param, kernel='linear'):
    '''Input:
    X1, X2: collections of points on which to compute the Gram matrix
    kernel: can be 'linear', 'polynomial' or 'gaussian'
    param: is [] for the linear kernel, the exponent of the polynomial kernel,
           or the variance for the gaussian kernel

    Output:
    k: Gram matrix'''
    if kernel == 'linear':
        k = np.dot(X1, np.transpose(X2))
    elif kernel == 'polynomial':
        k = np.power((1 + np.dot(X1, np.transpose(X2))), param)
    elif kernel == 'gaussian':
        k = np.exp(float(-1) / float((2 * param ** 2)) * sqDist(X1, X2))
    return k
