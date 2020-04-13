import numpy as np
# import jax.numpy as np
from math import pi, e, sqrt, exp
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from tqdm import tqdm


def multi_gaussian(x, mean, var):
    p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
        -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
    return p

def derivative_measure(x, alpha):
    '''
    @x: robot state (x1, x2, theta)
    @alpha: landmark state (alpha1, alpha2)
    '''
    dm = np.zeros((2,2))
    dm[0,0] = 0.5 / sqrt((alpha[0]-x[0])**2 + (alpha[1]-x[1])**2) * 2 * (alpha[0]-x[0])
    dm[0,1] = 0.5 / sqrt((alpha[0]-x[0])**2 + (alpha[1]-x[1])**2) * 2 * (alpha[1]-x[1])
    dm[1,0] = -1 / ( 1 + (alpha[1]-x[1])**2 / (alpha[0]-x[0])**2 ) * (alpha[1]-x[1]) /         (alpha[0]-x[0])**2
    dm[1,1] = 1 / ( 1 + (alpha[1]-x[1])**2 / (alpha[0]-x[0])**2 ) * (1) / (alpha[0]-x[0])
    return dm

def fisher_mat(x, alpha, cov_inv):
    '''
    @cov_inv: inverse of measurement noise covariance matrix
    '''
    dist = sqrt( (x[0]-alpha[0])**2 + (x[1]-alpha[1])**2 )
    if dist > 4:
        return np.zeros((2,2))
    else:
        dm = derivative_measure(x, alpha)
        return np.dot(np.dot(dm.T, cov_inv), dm)

def fisher_mat_expectation(x, alpha, cov_inv):
    '''
    @cov_inv: inverse of measurement noise covariance matrix
    '''
    dm = derivative_measure(x, alpha)
    return np.dot(np.dot(dm.T, cov_inv), dm)

def sample_expectation(rv, mean, cov, num=1000):
    val = 0
    sx, sy = np.random.multivariate_normal(mean, cov, num).T
    for i in range(sx.shape[0]):
        val += rv(sx[i], sy[i])
    val /= num
    return val

if __name__ == '__main__':
    # initialization
    means = np.load('/home/msun/Code/ErgodicBSP/test/mean_mat.npy')
    raw_cov = np.load('/home/msun/Code/ErgodicBSP/test/cov_mat.npy')
    mcov = np.eye(2) * 0.04
    mcov_inv = np.linalg.inv(mcov)

    # initialize meshgrid
    num_pts = 50
    area = 20
    dim = 2
    temp_grid = np.meshgrid(*[np.linspace(0, area, num_pts) for _ in range(dim)])
    grid = np.array([temp_grid[0].ravel(), temp_grid[1].ravel()]).T

    # calculate spatial distribution
    num_state = 3
    num_lm = int((means.shape[0]-num_state) / 2)
    vals = np.zeros(grid.shape[0])

    for i in tqdm(range(grid.shape[0])):  # iterate every grid in search space
        x = grid[i,:]
        fisher_mat_val = 0
        subfunc = lambda lm1,lm2: fisher_mat(x, [lm1,lm2], mcov_inv)
        for j in range(num_lm):     # iterate every "observed" landmark
            lm = means[2 + 2*j + 1: 2 + 2*j +3]
            cov = raw_cov[2 + 2*j + 1: 2 + 2*j + 3, 2 + 2*j + 1: 2 + 2*j +3]
            # cov_inv = np.linalg.inv(cov)
            if lm[0] !=0 and lm[1] != 0: # observed
                mat = sample_expectation(lambda lm1,lm2: fisher_mat(x, [lm1,lm2], mcov_inv), lm, cov, 1000)
            else: # unobserved landmark
                mat = np.zeros((2,2))
            fisher_mat_val += np.linalg.det(mat)
        vals[i] = fisher_mat_val
    vals = np.array(vals)
    vals /= np.sum(vals)

    # visualization
    xy = []
    for g in grid.T:
        xy.append(
            np.reshape(g, newshape=(num_pts,num_pts))
        )
    grid_vals = vals.reshape(num_pts, num_pts)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal', 'box')
    ax.contourf(*xy, grid_vals, levels=30)
    plt.show()

