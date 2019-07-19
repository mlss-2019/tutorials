import numpy as np
from regularizedKernLSTest import regularizedKernLSTest


def separatingFKernRLS(c, Xtr, Ytr, kernel, sigma, Xte, Yte, axs):
    '''The function classifies points evenly sampled in a visualization area,
    according to the classifier Regularized Least Squares

    Arguments:
    c: model weights
    Xtr: training input
    Ytr: training output
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    sigma: width of the gaussian kernel, if used
    Xte: test input
    Yte: test output

    Example of usage:

    from regularizationNetworks import MixGauss
    from regularizationNetworks import separatingFKernRLS
    from regularizationNetworks import regularizedKernLSTrain
    import numpy as np

    lam = 0.01
    kernel = 'gaussian'
    sigma = 1

    Xtr, Ytr = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.25'), 100)
    Xts, Yts = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.3'), 100)

    c = regularizedKernLSTrain.regularizedkernlstrain(Xtr, Ytr, 'gaussian', sigma, lam)
    separatingFKernRLS.separatingfkernrls(c, Xtr, Ytr, 'gaussian', sigma, Xts, Yts)
    '''

    step = 0.05

    x = np.arange(Xte[:, 0].min(), Xte[:, 0].max(), step)
    y = np.arange(Xte[:, 1].min(), Xte[:, 1].max(), step)

    xv, yv = np.meshgrid(x, y)

    xv = xv.flatten('F')
    xv = np.reshape(xv, (xv.shape[0], 1))

    yv = yv.flatten('F')
    yv = np.reshape(yv, (yv.shape[0], 1))

    xgrid = np.concatenate((xv, yv), axis=1)

    ygrid = regularizedKernLSTest(c, Xtr, kernel, sigma, xgrid)

    colors = [-1, +1]
    cc = []
    for item in Ytr:
        cc.append(colors[(int(item)+1)//2])

    axs.scatter(Xte[:, 0], Xte[:, 1], c=Yte, s=50)

    z = np.asarray(np.reshape(ygrid, (y.shape[0], x.shape[0]), 'F'))
    axs.contour(x, y, z, 1, colors='black')
