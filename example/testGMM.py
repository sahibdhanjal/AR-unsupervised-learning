from math import *
import numpy as np
import ggmm.gpu as ggmm

X = np.loadtxt('data.txt')

# N - training examples
# D - data dimension
# K - number of GMM components
N, D = X.shape
K = 3

ggmm.init()
gmm = ggmm.GMM(K,D)

thresh = 1e-1 # convergence threshold
n_iter = 20 # maximum number of EM iterations
init_params = 'wmc' # initialize weights, means, and covariances

# train GMM
gmm.fit(X, thresh, n_iter, init_params=init_params, verbose=True)

# retrieve parameters from trained GMM
weights = gmm.get_weights()
means = gmm.get_means()
covars = gmm.get_covars()

# compute posteriors of data
posteriors = gmm.compute_posteriors(X)

# print means
for i in means:
    # round to 2 decimal precision and print
    print(str(round(i[0], 2)) + ' ' + str(round(i[1], 2)))
