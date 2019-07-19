import scipy.io as sio
import numpy as np
from flipLabels import *


def two_moons(npoints, pflip):
    mat_contents = sio.loadmat('./moons_dataset.mat')
    Xtr = mat_contents['Xtr']
    Ytr = mat_contents['Ytr']
    Xts = mat_contents['Xts']
    Yts = mat_contents['Yts']
    npoints = min([100, npoints])
    i = np.random.permutation(100)
    sel = i[0:npoints]
    Xtr = Xtr[sel, :]
    if pflip > 1:
        Ytrn = flipLabels(Ytr[sel], pflip)
        Ytsn = flipLabels(Yts, pflip)
    else:
        Ytrn = np.squeeze(Ytr[sel])
        Ytsn = np.squeeze(Yts)
    return Xtr, Ytrn, Xts, Ytsn
