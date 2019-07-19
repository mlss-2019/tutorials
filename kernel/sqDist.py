import numpy as np


def sqDist(X1, X2):
    """Computes all the distances between two set of points stored in two matrices

    usage: D = sqDist(X1, X2)

    Arguments:
    X1: a matrix of size [n1xd], where each row is a d-dimensional point

    X2: a matrix of size [n2xd], where each row is a d-dimensional point

    Returns:
    D: a [n1xn2] matrix where each element (D)_ij is the distance between points (X_i, X_j)
    """
    sqx = np.sum(np.multiply(X1, X1), 1)
    sqy = np.sum(np.multiply(X2, X2), 1)
    return np.outer(sqx, np.ones(sqy.shape[0])) + np.outer(np.ones(sqx.shape[0]), sqy.T) - 2 * np.dot(X1, X2.T)
