"""
@author: luigi carratino
web: luigicarratino.com
"""

import numpy as np
from FALKON import FALKON


def GridSearchCV_FALKON(x, y, kernel, lam_list, kerpar_list, m_list, err_func, falkon_iter=5, perc=20, nrip=5, verbose=True):
    '''
    Input:
    x: the training examples
    y: the training labels
    kernel: the kernel function (it needs to be a function that given the kernel parameter returns a kernel function that takes two sets of points and returns the kernel matrix between the two sets. Example: lambda sigma: lambda A,B: rbf_kernel(A,B, gamma=1./(2*sigma**2)) ).
    lam_list: list of regularization parameters
        for example intlambda = np.array([5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001])
    kerpar_list: list of kernel parameters
        for example intkerpar = np.array([10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01])
    m_list: list of values M
        for example intmpar = np.array([5,10,100,500,1000])
    err_func: function to compute the error of the learned functions (Examples: c_err_func = lambda Ypred, Ytrue: np.mean(np.sign(Ypred) != np.sign(Ytrue)))
    falkon_iter: maximum number of iterations of FALKON
    perc: percentage of the dataset to be used for validation, must be in range [1,100]
    nrip: number of repetitions of the test for each couple of parameters
    
    Returns:
    l, s, m: the lambda, kernel parameter and M value that minimize the median of the validation error
    vm, vs: median and variance of the validation error for each couple of parameters
    tm, ts: median and variance of the error computed on the training set for each couple of parameters

    Example of usage:

    import numpy as np
    from FALKON import FALKON
    from sklearn.metrics.pairwise import rbf_kernel
    from MixGauss import MixGauss

    Xtr, Ytr = MixGauss([[0;0],[1;1]],[0.5,0.25],100);

    kerpar_list = [2, 4, 6]
    lam_list = [1e-3, 1e-1, 1]
    m_list = [10, 100, 1000]
    kernel = lambda sigma: lambda A,B: rbf_kernel(A,B, gamma=1./(2*sigma**2))
    c_err_func = lambda Ypred, Ytrue: np.mean(np.sign(Ypred) != np.sign(Ytrue))

    best_l, best_s, best_m, vm, vs, tm, ts = GridSearchCV_FALKON(Xtr, Ytr, kernel, lam_list, kerpar_list, m_list, c_err_func)
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

    if isinstance(m_list, int):
        m_list = np.array([m_list])
    else:
        m_list = np.array(m_list)
    nm = m_list.size

    n = x.shape[0]
    ntr = int(np.ceil(n * (1 - float(perc) / 100)))

    tm = np.zeros((nlambda, nkerpar, nm))
    ts = np.zeros((nlambda, nkerpar, nm))
    vm = np.zeros((nlambda, nkerpar, nm))
    vs = np.zeros((nlambda, nkerpar, nm))

    # ym = float(y.max() + y.min()) / float(2)
    tt = 0
    il = 0
    for l in lam_list:
        iss = 0
        for s in kerpar_list:
            imm = 0
            for m in m_list:
                trerr = np.zeros((nrip, 1))
                vlerr = np.zeros((nrip, 1))
                if verbose:
                    print(f'Processing parameters conf {tt+1} out of {len(lam_list)*len(kerpar_list)*len(m_list)}')
                for rip in range(nrip):
                    i = np.random.permutation(n)
                    xtr = x[i[:ntr]]
                    ytr = y[i[:ntr]]
                    xvl = x[i[ntr:]]
                    yvl = y[i[ntr:]]

                    ker_f = kernel(s)
                    alpha, C = FALKON(xtr, ytr, m, ker_f, l, falkon_iter)                    
                    ypr_tr = ker_f(xtr, C).dot(alpha)
                    trerr[rip] = err_func(ypr_tr, ytr)
                    ypr_vl = ker_f(xvl, C).dot(alpha)
                    vlerr[rip] = err_func(ypr_vl, yvl)
                    #print('l: ', l, ' s: ', s, ' valErr: ', vlerr[rip], ' trErr: ', trerr[rip])
                tm[il, iss, imm] = np.median(trerr)
                ts[il, iss, imm] = np.std(trerr)
                vm[il, iss, imm] = np.median(vlerr)
                vs[il, iss, imm] = np.std(vlerr)
                tt = tt + 1
                imm = imm + 1
            iss = iss + 1
        il = il + 1
    idx_a, idx_b, idx_c = np.where(vm == np.amin(vm))
    best_l = lam_list[idx_a]
    best_s = kerpar_list[idx_b]
    best_m = m_list[idx_c]
    if verbose:
        print('Done!')

    return [best_l, best_s, best_m, vm, vs, tm, ts]
