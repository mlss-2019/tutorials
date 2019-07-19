import numpy as np


def MixGauss(means, sigmas, n):
    """Generates a 2D dataset of two classes with gaussian noise

    usage: X, Y = MixGauss(means, sigmas, n)

    Arguments:
    means: (size [pxd]) and should be of the form [m1, ... ,mp] (each mi is
    d-dimensional)

    sigmas: (size px1) should be in the form [sigma_1, ..., sigma_p]

    n: number of points per class

    X: obtained input data matrix (size [p*nxd])
    Y: obtained output data vector (size [p*n])

    Returns:
    X: A [p*nxd] matrix with the coordinates of each point
    Y: A [n] array with the labels of each point, the label is an integer
        from the interval [0,p-1]

    EXAMPLE: X, Y = MixGauss([[0,1],[0,1]],[0.5, 0.25],1000)
        generates a 2D dataset with two classes, the first one centered on (0,0)
        with standard deviation 0.5, the second one centered on (1,1)
        with standard deviation 0.25.
        Each class will contain 1000 points.

    to visualize: plt.scatter(X[:,1],X[:,2],s=25,c=Y)"""

    means = np.array(means)
    sigmas = np.array(sigmas)

    d = means.shape[1]
    num_classes = sigmas.size
    data = np.full((n * num_classes, d), np.inf)
    labels = np.zeros(n * num_classes)

    for idx, sigma in enumerate(sigmas):
        data[idx * n:(idx + 1) * n] = np.random.multivariate_normal(mean=means[idx], cov=np.eye(d) * sigmas[idx] ** 2,
                                                                    size=n)
        labels[idx * n:(idx + 1) * n] = idx

    return data, labels
