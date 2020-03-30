import autograd.numpy as np
from math import pi


mean = np.load("mean_mat.npy")
cov = np.load("cov_mat.npy")

def single_gaussian(x, mean, var):
    p = 1 / np.sqrt(2 * pi * var) * np.exp(-0.5 * (x - mean)**2 / var)
    return p

def multi_gaussian(x, mean, var):
    p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
        -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
    return p

