'''
Author : Sahib Dhanjal < dhanjalsahib@gmail.com >

Parameters:
    K       : number of GMM Components
    D       : data dimension
    weights : weights of individual components
    scale   : range of each dimension + range of covar
    samp    : scale points or not
    -------------------------------------------------------------------------
    for example, for D = 3, scale is a D+1 x 1 array which looks as follows
    [x_scale, y_scale, z_scale, sig_scale]. Points range for 0 - x_scale in
    x, and likewise in y and z. sig_scale measures the scale in which the
    covariance matrix for each Gaussian varies.
    -------------------------------------------------------------------------

Functions:
    load_training_data(K, D, weights, scale, file, plot)    : generates weighted points
    plotPoints(points)                                      : plots the weighted points
    resample(points, wghts)                                 : resample points based on wghts

'''
from random import randint
from scipy.stats import multivariate_normal as mvn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# resampling algo - http://cecas.clemson.edu/~ahoover/ece854/lecture-notes/lecture-pf.pdf
def resample(points, wghts):
    N, D = points.shape

    # normalize data
    wghts = a = np.divide(wghts, sum(wghts))
    output = np.zeros(shape=(N,D))

    # resample
    Q = np.cumsum(wghts)
    T = np.random.random(N+1)
    T.sort()
    T[N] = 1

    i = j = 1
    idx = [0]*N

    while(i<N):
        if (T[i]<Q[j]):
            idx[i] = j
            i = i+1
        else:
            j = j+1

    for i in range(N):
        output[i] = points[idx[i]]

    return output

def plotPoints(points):
    N,D = points.shape
    D = D-1
    if D==1:
        plt.plot(points[:, [0]], points[:, [1]])
        plt.xlabel('points - x')
        plt.ylabel('weights')
        plt.show()

    elif D==2:
        x, y, z = points[:,[0]].reshape(1,N), points[:, [1]].reshape(1,N), points[:, [2]].reshape(1,N)
        df = pd.DataFrame({'x': x[0], 'y': y[0], 'z': z[0]})

        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    elif D==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = points[:, [0]].reshape(1,N)[0]
        y = points[:, [1]].reshape(1,N)[0]
        z = points[:, [2]].reshape(1,N)[0]
        c = points[:, [3]].reshape(1,N)[0]
        sp = ax.scatter(x, y, z, c=c, cmap=plt.hot())
        plt.colorbar(sp)
        plt.show()

    else:
        raise Exception("Plotter cannot visualize data higher than 3 Dimensions")

def load_training_data(K, D, weights=[], scale=[], file="", plot=False, samp=False):
    # if weights are empty, initialize all weights to be equal
    if len(weights) == 0:
        weights = (1/K)*np.ones(K)

    # if scale array is empty, initialize all scales to be in range 0 - 100
    if len(scale) == 0:
        scale = 100*np.ones(D+1)

    idx = 0                                         # index counter
    N = 1                                           # number of data points
    means = np.zeros(shape=(K,D), dtype=int)        # matrix for means
    covars = np.zeros(shape=(D*K,D), dtype=int)     # matrix for covariances

    # calculate number of data points to be generated
    for i in range(D):
        N *= scale[i]

    # initialize K gaussian means/covars
    for i in range(K):
        # choose point at random to center gaussian around
        point = []
        for j in range(D):
            point.append(randint(1,scale[j]))
        sig = randint(1,scale[-1])

        # set means and covariances
        means[i][:] = point
        covars[D*i:D*i+D][:] = sig*np.eye(D)

    # initialize N points
    div = 1 ; mod = 1
    points = np.zeros(shape=(N,D))
    for j in range(D-1, -1, -1):
        mod *= scale[j]
        for i in range(N):
            points[i][j] = (i%mod)//div
        div *= scale[j]

    # calculate weights based on Gaussian Distribution
    wghts = np.zeros(N)
    for i in range(K):
        wghts = wghts + weights[i]*mvn.pdf(points, means[i][:], covars[D*i:D*i+D][:])
    print(sum(wghts))

    # print means and covariances
    print("GMM generated around:\n")
    for i in range(K):
        print "Point",i+1,":"
        print "mean:[", ",".join(str(j) for j in means[i]), "]"
        print "weight:",weights[i]
        print "covar:"
        print covars[D*i:D*i+D][:]
        print "========================="

    # concatenate wghts with points
    weighted_points = np.zeros(shape=(N,D+1))
    for i in range(N):
        weighted_points[i][:] = np.append(list(map(int,points[i][:])), wghts[i])

    # plot the points if requested
    if plot:
        try:
            plotPoints(weighted_points)
        except Exception as err:
            print(err)

    # resample points based on weights
    if samp:
        output = resample(points, wghts)
    else:
        output = weighted_points

    # write to data file if filename given
    if file!="":
        # write data to file
        f = open(file, "w")
        N, _ = output.shape
        for i in range(N):
            s = " ".join(str(j) for j in output[i]) + "\n"
            f.write(s)
        f.close()

    return output
