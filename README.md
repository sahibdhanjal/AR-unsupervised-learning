==============================================================
Weighted Expectation Maximization for Gaussian Mixture Models
==============================================================

Python module to train weighted GMMs using CUDA (via CUDAMat)

### Contents
[Dependencies](#dependencies)
[Installation](#installation)
[Test Generation](#test-generation)
[Example usage](#example-usage)
[Documentation](#documentation)

### Dependencies

* Not Windows (only tested on Linux and Mac)
* CUDA 6.0+ (only tested with 6.0)
* numpy
* CUDAMat, avaiable here: https://github.com/cudamat/cudamat.git
* future: http://python-future.org/index.html
* nose (optional, for running tests)

### Installation

Clone wgmm and CUDAMat in local install path:
```bash
cd ${INSTALL_PATH}
git clone https://github.com/sahibdhanjal/Weighted-Expectation-Maximization.git
git clone https://github.com/cudamat/cudamat.git
```

Compile and install CUDAMat:
```bash
cd ${INSTALL_PATH}/cudamat
sudo python setup.py install
```
Run CUDAMat tests (optional, requires nose):
```bash
cd ${INSTALL_PATH}/cudamat
nosetests
```
Run wgmm tests (optional, requires nose):
```bash
cd ${INSTALL_PATH}/wgmm
nosetests
```
Install wgmm:
```bash
cd ${INSTALL_PATH}/wgmm
sudo pip install .
```

### Test Generation
1. Open the example folder
2. Open a new terminal and run ```bash python testGen.py ```

### Example Usage

```python
import wgmm.gpu as wgmm

X = some_module.load_training_data()

# N - training examples
# D - data dimension
# K - number of GMM components
N, D = X.shape
K = 128

wgmm.init()
gmm = wgmm.GMM(K,D)

thresh = 1e-3 # convergence threshold
n_iter = 20 # maximum number of EM iterations
init_params = 'wmc' # initialize weights, means, and covariances

# train GMM
gmm.fit(X, thresh, n_iter, init_params=init_params)

# retrieve parameters from trained GMM
weights = gmm.get_weights()
means = gmm.get_means()
covars = gmm.get_covars()

# compute posteriors of data
posteriors = gmm.compute_posteriors(X)
```

For testing it on randomly generated data, generate test case as described above and then run ```bash python gmmtest.py```

### Documentation
Documentation available [here](http://ebattenberg.github.io/wgmm)
