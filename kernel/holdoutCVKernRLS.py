import numpy as np
from regularizedKernLSTrain import *
from regularizedKernLSTest import *
from calcErr import *


def holdoutCVKernRLS(x, y, perc, nrip, kernel, lam_list, kerpar_list):
    '''
    Input:
    xtr: the training examples
    ytr: the training labels
    kernel: the kernel function (see KernelMatrix documentation).
    perc: percentage of the dataset to be used for validation, must be in range [1,100]
    nrip: number of repetitions of the test for each couple of parameters
    lam_list: list of regularization parameters
        for example intlambda = np.array([5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001])
    kerpar_list: list of kernel parameters
        for example intkerpar = np.array([10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01])

    Returns:
    l, s: the couple of lambda and kernel parameter that minimize the median of the validation error
    vm, vs: median and variance of the validation error for each couple of parameters
    tm, ts: median and variance of the error computed on the training set for each couple of parameters

    Example of usage:

    from regularizationNetworks import MixGauss
    from regularizationNetworks import holdoutCVKernRLS
    import matplotlib.pyplot as plt
    import numpy as np

    lam_list = np.array([5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001])
    kerpar_list = np.array([10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01])
    xtr, ytr = MixGauss.mixgauss([[0;0],[1;1]],[0.5,0.25],100);
    l, s, Vm, Vs, Tm, Ts = holdoutCVKernRLS.holdoutcvkernrls(xtr, ytr,'gaussian', 0.5, 5, lam_list, kerpar_list);
    plt.plot(lam_list, vm, 'b')
    plt.plot(lam_list, tm, 'r')
    plt.show()
    '''

    if perc < 1 or perc > 100:
        print("p should be a percentage value between 0 and 100.")
        return -1

    if isinstance(kerpar_list, int):
        kerpar_list = np.array([kerpar_list])
    else:
        kerpar_list = np.array(kerpar_list)
    nkerpar = kerpar_list.size

    if isinstance(lam_list, int):
        lam_list = np.array([lam_list])
    else:
        lam_list = np.array(lam_list)
    nlambda = lam_list.size

    n = x.shape[0]
    ntr = int(np.ceil(n * (1 - float(perc) / 100)))

    tm = np.zeros((nlambda, nkerpar))
    ts = np.zeros((nlambda, nkerpar))
    vm = np.zeros((nlambda, nkerpar))
    vs = np.zeros((nlambda, nkerpar))

    ym = float(y.max() + y.min()) / float(2)

    il = 0
    for l in lam_list:
        iss = 0
        for s in kerpar_list:
            trerr = np.zeros((nrip, 1))
            vlerr = np.zeros((nrip, 1))
            for rip in range(nrip):
                i = np.random.permutation(n)
                xtr = x[i[:ntr]]
                ytr = y[i[:ntr]]
                xvl = x[i[ntr:]]
                yvl = y[i[ntr:]]

                w = regularizedKernLSTrain(xtr, ytr, kernel, s, l)
                trerr[rip] = calcErr(regularizedKernLSTest(w, xtr, kernel, s, xtr), ytr, ym)
                vlerr[rip] = calcErr(regularizedKernLSTest(w, xtr, kernel, s, xvl), yvl, ym)
                #print('l: ', l, ' s: ', s, ' valErr: ', vlerr[rip], ' trErr: ', trerr[rip])
            tm[il, iss] = np.median(trerr)
            ts[il, iss] = np.std(trerr)
            vm[il, iss] = np.median(vlerr)
            vs[il, iss] = np.std(vlerr)
            iss = iss + 1
        il = il + 1
    row, col = np.where(vm == np.amin(vm))
    l = lam_list[row]
    s = kerpar_list[col]

    return [l, s, vm, vs, tm, ts]
