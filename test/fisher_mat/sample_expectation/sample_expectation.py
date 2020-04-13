import numpy as np

# initialize parameters
num_pts = 100
mean = np.array([1., 2.])
sig = 1.

# sample data
sample = np.random.randn(num_pts, 2) * sig
sample += mean

# calculate expectation
expectation = np.array([np.sum(sample[:,0]), np.sum(sample[:,1])]) / num_pts
print(expectation)
