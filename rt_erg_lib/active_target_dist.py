# import rospy

import numpy as np
import numpy.random as npr
from math import pi
from .utils import *
from scipy.stats import multivariate_normal


class TargetDist(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    # 2020-03-01: add "size" parameter to support customizable exploration area size
    def __init__(self, num_pts, means, vars, size):

        # TODO: create a message class for this
        # rospy.Subscriber('/target_distribution',  CLASSNAME, self.callback)

        self.num_pts = num_pts
        self.size = size
        grid = np.meshgrid(*[np.linspace(0, size, num_pts) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        # self.means = [npr.uniform(0.2, 0.8, size=(2,))
        #                     for _ in range(num_nodes)]
        # self.vars  = [npr.uniform(0.05, 0.2, size=(2,))**2
        #                     for _ in range(num_nodes)]
        self.means = means
        self.vars = vars

        # print("means: ", self.means)

        self.has_update = False
        self.grid_vals = self.__call__(self.grid)
        self.target_grid_vals = self.__call__(self.grid)
        self.belief_vals = np.ones(self.grid_vals.shape) / np.sum(np.ones(self.grid_vals.shape))

    def get_grid_spec(self, vals=None):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(self.num_pts, self.num_pts))
            )
        if vals is None:
            return xy, self.grid_vals.reshape(self.num_pts, self.num_pts)
        else:
            return xy, vals.reshape(self.num_pts, self.num_pts)

    def __call__(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.zeros(x.shape[0])
        for m, v in zip(self.means, self.vars):
            innerds = np.sum((x - m) ** 2 / v, 1)
            val += np.exp(-innerds / 2.0) / np.sqrt((2 * np.pi) ** 2 * np.prod(v))
        # normalizes the distribution
        val /= np.sum(val)
        return val

    def reset(self):
        self.grid_vals = self.__call__(self.grid)

    def multi_gaussian(self, x, mean, var):
        p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
            -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
        return p

    def update_intuitive(self, nStates, nLandmark, observed_table, belief_means, belief_vars, threshold=1e-3):
        '''
        intuitive update: using landmark uncertainty directly
        '''
        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
        grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
        vals = np.zeros(grid.shape[0])
        for i in range(nLandmark):
            if observed_table[i] == 1:
                mean = belief_means[nStates + 2 * i: nStates + 2 * i + 2]
                var = belief_vars[nStates + 2 * i: nStates + 2 * i + 2, nStates + 2 * i: nStates + 2 * i + 2]
                rv = multivariate_normal(mean, var)
                vals += rv.pdf(grid)
            else:
                pass

        if np.sum(vals) != 0:
            vals /= np.sum(vals)
        self.belief_vals = vals
        self.grid_vals = self.belief_vals

    def update_fim(self, nStates, nLandmark, observed_table, belief_means, belief_cov, mcov_inv, threshold=1e-3):
        '''
        update using fisher information matrix approximated at mean
        '''
        p = np.linalg.det(belief_cov[0: nStates, 0: nStates])

        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
        grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

        lm_cov_table = []
        for i in range(nLandmark):
            if observed_table[i] == 1:
                lm = belief_means[2 + 2*i + 1: 2 + 2*i + 3]
                cov = belief_cov[2+2*i+1:2+2*i+3, 2+2*i+1:2+2*i+3]
                lm_cov_table.append(multi_gaussian(lm, lm, cov))
            else:
                lm_cov_table.append(0)

        vals = np.zeros(grid.shape[0])

        for i in range(nLandmark):
            if observed_table[i] == 1:
                lm = belief_means[2 + 2*i + 1 : 2 + 2*i + 3]
                fish_mat_det = fisher_mat_broadcast(grid, lm, mcov_inv) #* lm_cov_table[i]**2
                vals += fish_mat_det

                '''
                # test new function for sample-based fim
                lcov = belief_cov[2+2*i+1:2+2*i+3, 2+2*i+1:2+2*i+3]
                dummy = fisher_mat_expectation_broadcast(grid, lm, mcov_inv, lcov)
                vals += dummy
                '''
            else:
                pass

        if np.sum(vals) != 0:
            vals /= np.sum(vals)
        # else:
        #     vals = self.target_grid_vals
        self.belief_vals = vals

        self.grid_vals = self.belief_vals

    def update_df(self, nStates, nLandmark, observed_table, belief_means, belief_cov, threshold=1e-3):
        '''
        update using uncertainty distance field (max)
        '''
        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
        grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

        state = []
        lm_det = []
        for i in range(nLandmark):
            if observed_table[i] == 1:
                state.append(belief_means[2+2*i+1 : 2+2*i+3])
                cov = belief_cov[3+2*i:5+2*i, 3+2*i:5+2*i]
                lm_det.append(np.linalg.det(cov))
        state = np.array(state)
        lm_det = 1 / np.array(lm_det)[:, np.newaxis]

        grid_x = grid[:,0]
        grid_y = grid[:,1]
        state_x = state[:,0][:, np.newaxis]
        state_y = state[:,1][:, np.newaxis]
        diff_x = grid_x - state_x
        diff_y = grid_y - state_y
        dist_xy = 1 / np.sqrt(diff_x**2 + diff_y**2) * lm_det
        vals = dist_xy.max(axis = 0)
        if(np.sum(vals) != 0):
            vals /= np.sum(vals)
        self.grid_vals = vals

    def update_df_2(self, nStates, nLandmark, observed_table, belief_means, belief_cov, threshold=1e-3):
        '''
        update using uncertainty distance field (max)
        '''
        state = []
        lm_det = []
        for i in range(nLandmark):
            if observed_table[i] == 1:
                state.append(belief_means[2+2*i+1 : 2+2*i+3])
                cov = belief_cov[3+2*i:5+2*i, 3+2*i:5+2*i]
                lm_det.append(np.linalg.det(cov))
        state = np.array(state)
        lm_det = np.array(lm_det)

        vals = np.zeros(self.grid.shape[0])
        for i in range(self.grid.shape[0]):
            r = self.grid[i]
            for j in range(state.shape[0]):
                l = state[j]
                dist = np.sqrt((l[0]-r[0])**2 + (l[1]-r[1])**2)
                vals[i] += np.exp(-dist) * (-np.log(lm_det[j]))
        vals /= np.sum(vals)
        self.grid_vals = vals

    def update_df_3(self, nStates, nLandmark, observed_table, belief_means, belief_cov, threshold, pos):
        '''
        update using uncertainty distance field (max)
        '''
        state = []
        lm_det = []
        for i in range(nLandmark):
            if observed_table[i] == 1:
                state.append(belief_means[2+2*i+1 : 2+2*i+3])
                cov = belief_cov[3+2*i:5+2*i, 3+2*i:5+2*i]
                lm_det.append(np.linalg.det(cov))

        state.append([10., 10.])
        lm_det.append(1e-09)
        state.append([12., 12.])
        lm_det.append(1e-09)
        state.append([14., 14.])
        lm_det.append(1e-09)

        state = np.array(state)
        lm_det = np.array(lm_det)

        vals = np.zeros(self.grid.shape[0])
        for i in range(self.grid.shape[0]):
            r = self.grid[i]
            for j in range(state.shape[0]):
                l = state[j]
                dist = np.sqrt((l[0]-r[0])**2 + (l[1]-r[1])**2)
                vals[i] += np.exp(-dist) * (-np.log(lm_det[j]))
        vals /= np.sum(vals)
        self.grid_vals = vals

