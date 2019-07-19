import numpy as np
from  kernelMatrix import *


def regularizedKernLSTest(c, Xtr, kernel, sigma, Xte):
    '''
    Arguments:
    c: model weights
    Xtr: training input
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    sigma: width of the gaussian kernel, if used
    Xts: test points

    Returns:
    y: predicted model values

    Example of usage:

    from regularizationNetworks import regularizedKernLSTest
    y =  regularizedKernLSTest.regularizedkernlstest(c, Xtr, 'gaussian', 1, Xte)
    '''

    Ktest = kernelMatrix(Xte, Xtr, sigma, kernel)
    y = np.dot(Ktest, c)

    return y
