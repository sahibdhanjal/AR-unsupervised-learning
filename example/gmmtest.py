from math import *
import numpy as np
import generator
import ggmm.gpu as ggmm

##################################################
# generate points
##################################################
# X = generator.load_training_data(5, 3, weights= [0.8, 0.03, 0.07, 0.06, 0.04], scale=[10, 10, 10, 100], plot=True)

# or use an already existing data file consisting of
# N - training examples
# D - data dimension
# K - number of GMM components
X = np.loadtxt("data.txt")
N,D = X.shape
X = generator.resample(X[:,0:D-1],X[:,[-1]])
D = D-1
K = 3

##################################################
# write converted data to file
##################################################
'''
f = open('converted.txt','w')
for i in range(N):
    s = " ".join(str(j) for j in X[i]) + "\n"
    f.write(s)
f.close()
'''

ggmm.init()
gmm = ggmm.GMM(K,D)

thresh = 1e-10 # convergence threshold
n_iter = 1000000 # maximum number of EM iterations
init_params = 'wmc' # initialize weights, means, and covariances

# train GMM
gmm.fit(X, thresh, n_iter, init_params=init_params, verbose=False)

# retrieve parameters from trained GMM
weights = gmm.get_weights()
means = gmm.get_means()
covars = gmm.get_covars()

# compute posteriors of data
posteriors = gmm.compute_posteriors(X)

print(means)
print(weights)
