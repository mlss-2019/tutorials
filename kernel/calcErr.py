import numpy as np


def calcErr(t, y, m):
    vt = (t >= m).astype(int)
    vy = (y >= m).astype(int)

    err = float(np.sum(vt != vy))/float(y.shape[0])
    return err
