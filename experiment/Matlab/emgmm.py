'''
Function/API Imports:
    Expectation Maximization
    MATLAB API
'''
#!/usr/bin/env python
# EM imports
import sys, os, time
import scipy.io as si
import numpy as np
import ggmm.gpu as ggmm

# Matlab inteface imports
import matlab.engine

'''
Functions:
'''
def readEngine(file):
    try:
        X = si.loadmat(file)['output']
    except:
        raise('No data matrix available! See if .mat is generated or not')
    return X

def EMEngine(X, thresh_, n_iter_, timer, K_, init_='wmc'):
    N, D = X.shape
    K = K_

    start = time.time()

    # train gmm
    ggmm.init()
    gmm = ggmm.GMM(K,D)
    gmm.fit(X, thresh = thresh_, n_iter = n_iter_, init_params=init_, verbose=False, iterations=timer)

    # retrieve parameters
    weights = gmm.get_weights()
    means = gmm.get_means()
    covars = gmm.get_covars()
    posteriors = gmm.compute_posteriors(X)

    end = time.time() - start

    if timer:
        print 'Max Iterations:',n_iter_
        print 'Convergence Threshold:',thresh_
        print 'Exectution Time:', end
        print 'Number of Gaussians:', K
        print '==========================='

    return (means, weights, covars)

def writeEngine(file, means, weights, covars):
    si.savemat(file, dict(means=means, weights=weights, covars=covars))

def execute(input, output, thresh, n_iter, timer, k):
    X = readEngine(input)
    means, weights, covars = EMEngine(X, thresh, n_iter, timer, k)
    writeEngine(output, means, weights, covars)
    return 1

'''
Main Function:
'''
if __name__=='__main__':
    if len(sys.argv)==4:
        inp = sys.argv[1]
        out = sys.argv[2]
        n_iter = 10**int(sys.argv[3])

    elif len(sys.argv)==5:
        inp = sys.argv[1]
        out = sys.argv[2]
        n_iter = 10**int(sys.argv[3])
        thresh = 10**(-1*int(sys.argv[4]))

    elif len(sys.argv)==6:
        inp = sys.argv[1]
        out = sys.argv[2]
        n_iter = 10**int(sys.argv[3])
        thresh = 10**(-1*int(sys.argv[4]))
        timer = int(sys.argv[5])

    elif len(sys.argv)==7:
        inp = sys.argv[1]
        out = sys.argv[2]
        n_iter = 10**int(sys.argv[3])
        thresh = 10**(-1*int(sys.argv[4]))
        timer = int(sys.argv[5])
        k = int(sys.argv[6])

    else:
        inp = 'data.mat'
        out = 'results.mat'
        thresh = 1e-10
        n_iter = 10**6
        timer = 0
        k = 3

    while True:
        execute(inp, out, thresh, n_iter, timer, k)
