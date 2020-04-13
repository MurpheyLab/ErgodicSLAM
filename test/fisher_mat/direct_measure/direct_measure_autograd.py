import matplotlib.pyplot as plt
from tqdm import tqdm
from math import pi, exp, e, sqrt, log

import numpy as onp
import jax.numpy as np
from jax import grad, jacfwd
from scipy.integrate import quad, dblquad


var = np.eye(2) * 0.5
var_inv = np.linalg.inv(var)
const1 = 1 / sqrt(np.linalg.det(2 * pi * var))

def multi_gaussian(x, mean, var):
    p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
        -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
    return p

# def gaussian2D(x, mu):
#     return const1 * exp(-0.5 * ( var_inv[0,0]*(x[0]-mu[0])**2 + (var_inv[0,1]+var_inv[1,0])*(x[0]-mu[0])*(x[1]-mu[1]) + var_inv[1,1]*(x[1]-mu[1])**2 ))

def gaussian2D(x):
    '''
    @x[0:2]: x
    @x[2:4]: mu
    '''
    return const1 * e ** (-0.5 * ( var_inv[0,0]*(x[0]-x[2])**2 + (var_inv[0,1]+var_inv[1,0])*(x[0]-x[2])*(x[1]-x[3]) + var_inv[1,1]*(x[1]-x[3])**2 ))

def score(x):
    '''
    @x[0:2]: x
    @x[2:4]: mu
    '''
    return np.log( const1 * e ** (-0.5 * ( var_inv[0,0]*(x[0]-x[2])**2 + (var_inv[0,1]+var_inv[1,0])*(x[0]-x[2])*(x[1]-x[3]) + var_inv[1,1]*(x[1]-x[3])**2 )) )



if __name__ == '__main__':
    mean = np.array([0, 0]) # true mean of the underlying distribution

    dpx = grad(gaussian2D)
    # print(gaussian2D([1., 1., 1., 1.]))
    # print(dpx(np.array([1., 1., 1., 1.])))

    sdot = grad(score)
    # print(score([1., 1., 1., 1.]))
    # print(sdot(np.array([1., 1., 1., 1.])))

    area = 10
    num_pts = 50
    temp_grid = np.meshgrid(*[np.linspace(-area, area, num_pts) for _ in range(2)])
    grid = onp.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

    vals = np.zeros(grid.shape[0])
    for i in tqdm(range(grid.shape[0])):
        mu = grid[i, :]
        sdot_val = sdot(np.array([mean[0], mean[1], mu[0], mu[1]]))[2:4]
        fish_mat = np.outer(sdot_val, sdot_val) * gaussian2D(np.array([mean[0], mean[1], mu[0], mu[1]]))
