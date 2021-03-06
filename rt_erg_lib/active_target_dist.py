# import rospy

import numpy as np
import numpy.random as npr
from math import pi
from .utils import *
from scipy.stats import multivariate_normal as mvn


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

    def update_fim_2(self, nStates, nLandmark, observed_table, belief_means, belief_cov, mcov_inv, threshold=1e-3):
        # parse input
        num_pts = int( np.sqrt(self.grid.shape[0]) )
        imcov = mcov_inv
        landmark_mean = []
        landmark_cov = []
        for i in range(nLandmark):
            if observed_table[i] == 1:
                mean = belief_means[nStates + 2 * i: nStates + 2 * i + 2]
                cov = belief_cov[nStates + 2 * i: nStates + 2 * i + 2, nStates + 2 * i: nStates + 2 * i + 2]
                landmark_mean.append(mean)
                landmark_cov.append(cov)
            else:
                pass
        landmark_mean = np.array(landmark_mean)
        landmark_cov = np.array(landmark_cov)

        # compute fim
        vals = np.zeros(self.grid.shape[0])

        grid_x = self.grid[:, 0]
        grid_y = self.grid[:, 1]
        diff_x = grid_x - grid_x[:, np.newaxis]
        diff_y = grid_y - grid_y[:, np.newaxis]
        dist_xy = np.sqrt(diff_x**2 + diff_y**2)

        range_flag_1 = 0.0 < dist_xy
        range_flag_1 = range_flag_1.astype(int)
        range_flag_2 = dist_xy < 4.0
        range_flag_2 = range_flag_2.astype(int)
        range_flag = range_flag_1 * range_flag_2

        zero_flag = np.ones((num_pts**2, num_pts**2)) - np.eye(num_pts**2)

        dist_xy = dist_xy + 1e-09

        dm11 = -(grid_x - grid_x[:, np.newaxis]) / dist_xy * zero_flag
        dm12 = -(grid_y - grid_y[:, np.newaxis]) / dist_xy * zero_flag
        dm21 = (grid_y - grid_y[:, np.newaxis]) / dist_xy**2 * zero_flag
        dm22 = -(grid_x - grid_x[:, np.newaxis]) / dist_xy**2 * zero_flag

        fim11 = dm11 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm21 * (dm11*imcov[0,1] + dm21*imcov[1,1])
        fim12 = dm12 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm22 * (dm11*imcov[0,1] + dm21*imcov[1,1])
        fim21 = dm11 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm21 * (dm12*imcov[0,1] + dm22*imcov[1,1])
        fim22 = dm12 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm22 * (dm12*imcov[0,1] + dm22*imcov[1,1])

        det = fim11 * fim22 - fim12 * fim21
        det = det * range_flag

        for i in range(landmark_mean.shape[0]):
            distr = mvn.pdf(self.grid, landmark_mean[i], landmark_cov[i] * 50)
            scaled_det = det * distr[:, np.newaxis]
            det_vals = np.sum(scaled_det, axis=0)
            vals += det_vals

        vals_sum = np.sum(vals)
        if vals_sum != 0:
            vals /= vals_sum

        self.grid_vals = vals

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

    def update_df_3(self, nStates, nLandmark, observed_table, belief_means, belief_cov, threshold):
        '''
        update using uncertainty distance field (fast implementation)
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

        if len(state) == 0:
            print('no observation available.\n')
            return

        grid_x = self.grid[:, 0]
        grid_y = self.grid[:, 1]
        state_x = state[:, 0][:, np.newaxis]
        state_y = state[:, 1][:, np.newaxis]
        diff_x = grid_x - state_x
        diff_y = grid_y - state_y
        dist_xy = np.sqrt(diff_x**2 + diff_y**2)
        dist_xy = np.exp(-dist_xy).T * ( -np.log(lm_det) )
        vals = np.sum(dist_xy, axis=1)

        vals /= np.sum(vals)
        self.grid_vals = vals

    def update_df_4(self, nStates, nLandmark, observed_table, belief_means, belief_cov, threshold, new_observed):
        '''
        update using uncertainty distance field (frontier exploration)
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

        if len(state) == 0:
            print('no observation available.\n')
            return

        factor = np.ones(state.shape[0])
        if new_observed is not None:
            new_id = int( np.sum(observed_table[0:new_observed]) )
            factor[new_id] = 4

        grid_x = self.grid[:, 0]
        grid_y = self.grid[:, 1]
        state_x = state[:, 0][:, np.newaxis]
        state_y = state[:, 1][:, np.newaxis]
        diff_x = grid_x - state_x
        diff_y = grid_y - state_y
        dist_xy = np.sqrt(diff_x**2 + diff_y**2)
        dist_xy = np.exp(-dist_xy).T * ( -np.log(lm_det) ) * factor
        vals = np.sum(dist_xy, axis=1)

        vals /= np.sum(vals)
        self.grid_vals = vals

