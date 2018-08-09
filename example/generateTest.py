import numpy as np
from random import randint
from scipy.stats import multivariate_normal as mvn

'''
-------------------------------------------------------------------------
Parameters:
-------------------------------------------------------------------------
K                               : number of GMM Components
D                               : data dimension
pi                              : weights of individual components
scale                           : range of each dimension + range of covar
-------------------------------------------------------------------------
for example, for D = 3, scale is [x_scale, y_scale, z_scale, sigma_scale]
and points range for 0 - x_scale in x, and likewise in y and z
sigma_scale measures the scale in which the covariances  matrix for each
Gaussian varies.
-------------------------------------------------------------------------

-------------------------------------------------------------------------
Functions:
-------------------------------------------------------------------------
generatePoints(K, D, pi, scale) : generates weighted points
plotPoints(points)              : plots the weighted points
-------------------------------------------------------------------------
'''

def generatePoints(K, D, pi, filename='data.txt', x_scale = 100, y_scale = 100, z_scale = 100, sig_scale = 100):
    N = x_scale*y_scale*z_scale                     # training examples

    means = np.zeros(shape=(K,D), dtype=int)        # matrix for means
    covars = np.zeros(shape=(D*K,D), dtype=int)     # matrix for covariances

    # initialize K points
    for i in range(K):
        # choose point at random to center gaussian around
        x, y, z = randint(0,x_scale), randint(0,y_scale), randint(0,z_scale)
        sig = randint(0,sig_scale)
        # set means and covariances
        means[i][:] = [x,y,z]
        covars[3*i:3*i+3][:] = sig*np.eye(D)

    # initialize points
    points = np.zeros(shape=(N,D), dtype=int)
    ctr = 0
    for x in range(x_scale):
        for y in range(y_scale):
            for z in range(z_scale):
                points[ctr][:] = [x,y,z]
                ctr += 1

    # calculate weights
    weights = np.zeros(N)
    for i in range(K):
        weights = weights + pi[i]*mvn.pdf(points, means[i][:], covars[3*i:3*i+3][:])

    # update weights for points
    weighted_points = np.zeros(shape=(N,D+1))
    for i in range(N):
        x, y, z = map(int,points[i][:])
        weighted_points[i][:] = [x, y, z, weights[i]]

    f = open(filename, "w")
    for i in range(N):
        s = ""
        for x in range(D):
            s += str(int(weighted_points[i][x])) + " "
        s += str(weighted_points[i][D]) + '\n'
        f.write(s)
    f.close()

    print("Points Generated with Means/ Covariances/ Weights:")

    for i in range(K):
        print "Point",i,":"
        print "mean:", " ".join(str(j) for j in means[i])
        print "weight:",pi[i]
        print "covar:"
        print covars[3*i:3*i+3][:]




if __name__ == '__main__':
    D = 3                                           # dimension
    K = 3                                           # number of GMM components
    pi = [0.3, 0.6, 0.1]                            # weights of Gaussians
    generatePoints(K, D, pi)
