import sys
sys.path.append("/home/msun/Code/ErgodicBSP")
from rt_erg_lib.utils import *

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from numpy import pi, sqrt


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
    dm[1,0] = -1 / ( 1 + (alpha[1]-x[1])**2 / (alpha[0]-x[0])**2 ) * (alpha[1]-x[1]) / (alpha[0]-x[0])**2
    dm[1,1] = 1 / ( 1 + (alpha[1]-x[1])**2 / (alpha[0]-x[0])**2 ) * (1) / (alpha[0]-x[0])
    return dm

def derivative_measure2(x, landmark):
    '''
    @x: robot state (x1, x2), no theta
    @landmark: landmark state (l1, l2)
    '''
    dm = np.zeros((2,2))
    dm[0,0] = 1 / sqrt((landmark[0]-x[0])**2 + (landmark[1]-x[1])**2) * (-2) * (landmark[0]-x[0])
    dm[0,1] = 1 / sqrt((landmark[0]-x[0])**2 + (landmark[1]-x[1])**2) * (-2) * (landmark[1]-x[1])
    dm[1,0] = 1 / ( 1 + (landmark[1]-x[1])**2 / (landmark[0]-x[0])**2 ) * (landmark[1]-x[1]) / (landmark[0]-x[0])**2
    dm[1,1] = -1 / ( 1 + (landmark[1]-x[1])**2 / (landmark[0]-x[0])**2 ) * (1) / (landmark[0]-x[0])
    return dm


def range_bearing(robot, alpha):
    '''
    range-bearing measurement model
    '''
    delta = alpha - robot[0:2]
    rangee = np.sqrt(np.dot(delta.T, delta))
    bearing = math.atan2(delta[1], delta[0]) - robot[2]
    # todo: normalize bearing ?
    return np.array([rangee, bearing])


if __name__ == '__main__':
    # means = np.load('/home/msun/Code/ErgodicBSP/test/fisher_info/mean_mat.npy')
    # raw_vars = np.load('/home/msun/Code/ErgodicBSP/test/fisher_info/cov_mat.npy')

    means = np.load('/home/msun/Code/ErgodicBSP/test/mean_mat.npy')
    raw_vars = np.load('/home/msun/Code/ErgodicBSP/test/cov_mat.npy')

    mcov_inv = np.linalg.inv(np.diag([0.04**1, 0.04**1]))

    num_pts = 50
    size = 20
    temp_grid = np.meshgrid(*[np.linspace(0, size, num_pts) for _ in range(2)])
    grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
    print(grid.shape)

    num_state = 3
    num_landmark = int((means.shape[0] - num_state) / 2)
    init_table = np.zeros(num_landmark)
    varc = []
    for i in range(num_landmark):
        var = raw_vars[2 + 2*i + 1: 2 + 2*i + 3, 2 + 2*i + 1: 2 + 2*i +3]
        if var[0, 0] != 99999999:
            init_table[i] = 1
        else:
            init_table[i] = 0
        varc.append(var)
    varc = np.array(varc)

    robot = means[0: num_state]
    vals = np.zeros(grid.shape[0])

    for i in range(num_landmark):
        if init_table[i] == 1:
            lm = means[2 + 2*i + 1 : 2 + 2*i + 3]
            # fish_mat_det = fisher_mat_broadcast(grid, lm, mcov_inv) # * lm_cov_table[i]**2
            # vals += fish_mat_det
            # test new function
            lcov = raw_vars[2 + 2*i + 1 : 2 + 2*i + 3, 2 + 2*i + 1 : 2 + 2*i + 3]
            dummy = fisher_mat_expectation_broadcast(grid, lm, mcov_inv, lcov, s_num=1000)
            vals += dummy
        else:
            pass

    if np.sum(vals) != 0:
        vals /= np.sum(vals)

    # preprocess for visualzation
    xy = []
    for g in grid.T:
        xy.append(
            np.reshape(g, newshape=(num_pts, num_pts))
        )
    grid_vals = vals.reshape(num_pts, num_pts)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal', 'box')
    ax.contourf(*xy, grid_vals, levels=50)
    plt.show()

