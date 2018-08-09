from math import *
import numpy as np
import generator
import wgmm.gpu as wgmm

'''
load_training_data(K, D, weights, scale, file, plot)
K                   : number of GMM Components
D                   : data dimension
weights (optional)  : weights of each Gaussian in GMM. default weights are equal for all gaussians
scale (optional)    : range in which each dimension varies. default range is 100 in all dimensions
file (optional)     : filename on which data needs to be written. doesn't write to file is blank
plot (optional)     : bool flag to plot data or not
'''
##################################################
# generate points
##################################################
X = generator.load_training_data(5, 3, weights= [0.8, 0.03, 0.07, 0.06, 0.04], scale=[10, 10, 10, 100], plot=True)
# or use an already existing data file
# X = np.loadtxt('data.txt')

##################################################
# start weighted GMM fitting here
##################################################
# N - training examples
# D - data dimension
# K - number of GMM components
N, D = X.shape
K = 3

wgmm.init()
gmm = wgmm.GMM(K,D)

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
