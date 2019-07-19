"""
@author: luigi carratino
web: luigicarratino.com
"""

import numpy as np

def FALKON(X, Y, M, KernelMatrix, lam, t):
    n = X.shape[0]
    # select Nystrom centers uniformly at random
    nys_idxs = np.random.permutation(n)[0:M]
    C = X[nys_idxs, :]
    print('Computing preconditioner')
    KMM = KernelMatrix(C,C)
    T = np.linalg.cholesky(KMM + M*np.eye(M))
    A = np.linalg.cholesky(T.dot(T.T)/M + lam*np.eye(M))
        
    def KnMtimesVector(u, v):
        w = np.zeros((M,1))
        ms = np.ceil(np.linspace(0, n, np.ceil(n/M)+1)).astype(int)
        for i in range(int(np.ceil(n/M))):
            Kr = KernelMatrix(X[ms[i]:ms[i+1],:], C)
            w = w + Kr.T.dot((Kr.dot(u) + v[ms[i]:ms[i+1]]))
        return w

    def BHB(u):
        w = np.linalg.solve(A.T, (np.linalg.solve(T.T,(KnMtimesVector(np.linalg.solve(T, np.linalg.solve(A,u)), np.zeros((n,1)))/n)) + lam*np.linalg.solve(A,u)))
        return w
    
    def conjgrad(funA, r, tmax):
        eps = 1e-15
        p = r
        rsold = r.T.dot(r)
        beta = np.zeros((r.shape[0], 1))
        for i in range(tmax):
            print(f'Iteration {i+1} out of {tmax}')
            Ap = funA(p)
            a = rsold/(p.T.dot(Ap) + eps)
            beta = beta + a*p
            r = r - a*Ap
            rsnew = r.T.dot(r)
            p = r + (rsnew/rsold)*p
            rsold = rsnew
        return beta
    
    print('Starting CG iterations')
    print(KnMtimesVector(np.zeros((M,1)), Y/n).shape)
    r = np.linalg.solve(A.T, np.linalg.solve(T.T, KnMtimesVector(np.zeros((M,1)), Y/n)))
    beta = conjgrad(BHB, r, t);
    alpha = np.linalg.solve(T, np.linalg.solve(A,beta));
    return alpha, C