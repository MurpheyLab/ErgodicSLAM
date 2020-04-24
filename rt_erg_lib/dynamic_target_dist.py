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

    def get_grid_spec(self):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(self.num_pts, self.num_pts))
            )
        return xy, self.grid_vals.reshape(self.num_pts, self.num_pts)

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

    def update0(self, nStates, nLandmark, observed_table, belief_means, belief_vars):
        p = np.linalg.det(belief_vars[0: nStates, 0: nStates])
        print("p: ", p)
        threshold = 1e-05
        if p < threshold:
            print("normal exploration")
            self.reset()
        else:
            temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
            grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

            vals = np.zeros(grid.shape[0])
            for i in range(grid.shape[0]):
                for j in range(nLandmark):
                    if observed_table[j] == 1:
                        x = grid[i, :]
                        mean = belief_means[nStates + 2 * j: nStates + 2 * j + 2]
                        var = belief_vars[nStates + 2 * j: nStates + 2 * j + 2, nStates + 2 * j: nStates + 2 * j + 2]
                        p = self.multi_gaussian(x, mean, var)
                        vals[i] += p
                    else:
                        pass
            if np.sum(vals) != 0:
                vals /= np.sum(vals)
            # else:
            #    vals = np.ones(grid.shape[0])


            print("replanning")
            alpha = p / (p + threshold)
            # self.grid_vals = (1 - alpha) * self.grid_vals + alpha * vals
            self.grid_vals = vals

    def update1(self, nStates, nLandmark, observed_table, belief_means, belief_vars, threshold=1e-3):
        '''
        intuitive update: using landmark uncertainty directly
        '''
        p = np.linalg.det(belief_vars[0: nStates, 0: nStates])
        # print("\np: ", p)
        if p < threshold:
            self.grid_vals = self.target_grid_vals # replace with "hard" switch
            return 0

        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
        grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
        ''' old implementation (too slow!)
        vals = np.zeros(grid.shape[0])
        for i in range(grid.shape[0]):
            for j in range(nLandmark):
                if observed_table[j] == 1:
                    x = grid[i, :]
                    mean = belief_means[nStates + 2 * j: nStates + 2 * j + 2]
                    var = belief_vars[nStates + 2 * j: nStates + 2 * j + 2, nStates + 2 * j: nStates + 2 * j + 2]
                    px = self.multi_gaussian(x, mean, var)
                    vals[i] += px
                else:
                    pass
        '''
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
        # else:
        #     vals = np.ones(grid.shape[0])

        alpha = (p / (p + threshold)) ** 2
        # print("alpha: ", alpha)
        # self.grid_vals = (1 - alpha) * self.target_grid_vals + alpha * vals
        # self.grid_vals /= np.sum(self.grid_vals)
        self.grid_vals = vals # replace with "hard" switch

    def update2(self, nStates, nLandmark, observed_table, belief_means, belief_cov, mcov_inv, threshold=1e-3):
        '''
        update using fisher information matrix approximated at mean
        '''
        p = np.linalg.det(belief_cov[0: nStates, 0: nStates])
        # print("\np: ", p)
        if p < threshold:
            self.grid_vals = self.target_grid_vals
            return 0

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
        ''' old implementation (too slow !)
        for i in range(grid.shape[0]):
            x = grid[i,:]
            fisher_mat_val = 0
            # subfunc = lambda lm1, lm2: fisher_mat(x, [lm1, lm2], mcov_inv)
            for j in range(nLandmark):
                if observed_table[j] == 1:
                    lm = belief_means[2 + 2*j + 1: 2 + 2*j + 3]
                    cov = belief_cov[2+2*j+1:2+2*j+3, 2+2*j+1:2+2*j+3]
                    mat = fisher_mat(x, lm, mcov_inv) * lm_cov_table[j]
                else:
                    mat = np.zeros((nStates-1, nStates-1))
                fisher_mat_val += np.linalg.det(mat)
            vals[i] = fisher_mat_val
        vals = np.array(vals)
        '''
        for i in range(nLandmark):
            if observed_table[i] == 1:
                lm = belief_means[2 + 2*i + 1 : 2 + 2*i + 3]
                # fish_mat_det = fisher_mat_broadcast(grid, lm, mcov_inv) # * lm_cov_table[i]**2
                # vals += fish_mat_det

                # test new function for sample-based fim
                lcov = belief_cov[2+2*i+1:2+2*i+3, 2+2*i+1:2+2*i+3]
                dummy = fisher_mat_expectation_broadcast(grid, lm, mcov_inv, lcov)
                vals += dummy
            else:
                pass

        if np.sum(vals) != 0:
            vals /= np.sum(vals)
        # else:
        #     vals = np.ones(grid.shape[0])

        alpha = (p / (p + threshold)) ** 2
        # print("alpha: ", alpha)
        # self.grid_vals = (1 - alpha) * self.target_grid_vals + alpha * vals
        # self.grid_vals /= np.sum(self.grid_vals)
        self.grid_vals = vals

    def update3(self, nStates, nLandmark, observed_table, belief_means, belief_cov, threshold=1e-3):
        '''
        update using uncertainty distance field (sum)
        '''
        p = np.linalg.det(belief_cov[0: nStates, 0: nStates])
        # print("\np: ", p)

        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
        grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

        lm_det_table = []
        for i in range(nLandmark):
            if observed_table[i] == 1:
                lm = belief_means[2 + 2*i + 1: 2 + 2*i + 3]
                cov = belief_cov[2+2*i+1:2+2*i+3, 2+2*i+1:2+2*i+3]
                lm_det_table.append(np.linalg.det(cov))
            else:
                lm_det_table.append(0)

        vals = np.zeros(grid.shape[0])
        for i in range(grid.shape[0]):
            x = grid[i,:]
            for j in range(nLandmark):
                if observed_table[j] == 1:
                    lm = belief_means[2 + 2*j + 1: 2 + 2*j + 3]
                    cov = belief_cov[2+2*j+1:2+2*j+3, 2+2*j+1:2+2*j+3]
                    dist_val = 1 / ( sqrt( (x[0]-lm[0])**2 + (x[1]-lm[1])**2 ) * lm_det_table[j] )
                    vals[i] += dist_val
                else:
                    pass
        vals = np.array(vals)
        if np.sum(vals) != 0:
            vals /= np.sum(vals)
        # else:
        #     vals = np.ones(grid.shape[0])

        # threshold = 1e-03
        # threshold = 4e-04
        alpha = (p / (p + threshold)) ** 2
        # print("alpha: ", alpha)
        self.grid_vals = (1 - alpha) * self.target_grid_vals + alpha * vals

    def update4(self, nStates, nLandmark, observed_table, belief_means, belief_cov, threshold=1e-3):
        '''
        update using uncertainty distance field (max)
        '''
        p = np.linalg.det(belief_cov[0: nStates, 0: nStates])
        # print("\np: ", p)

        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
        grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

        lm_det_table = []
        for i in range(nLandmark):
            if observed_table[i] == 1:
                lm = belief_means[2 + 2*i + 1: 2 + 2*i + 3]
                cov = belief_cov[2+2*i+1:2+2*i+3, 2+2*i+1:2+2*i+3]
                lm_det_table.append(np.linalg.det(cov))
            else:
                lm_det_table.append(0)

        vals = np.zeros(grid.shape[0])
        for i in range(grid.shape[0]):
            x = grid[i,:]
            dist_vals = []
            for j in range(nLandmark):
                if observed_table[j] == 1:
                    lm = belief_means[2 + 2*j + 1: 2 + 2*j + 3]
                    cov = belief_cov[2+2*j+1:2+2*j+3, 2+2*j+1:2+2*j+3]
                    dist_val = 1 / ( sqrt( (x[0]-lm[0])**2 + (x[1]-lm[1])**2 ) * lm_det_table[j] )
                    dist_vals.append(dist_val)
                else:
                    dist_vals.append(0)
            vals[i] += max(dist_vals)
        vals = np.array(vals)
        if np.sum(vals) != 0:
            vals /= np.sum(vals)
        # else:
        #     vals = np.ones(grid.shape[0])

        # threshold = 1e-03
        # threshold = 4e-04
        alpha = (p / (p + threshold)) ** 2
        # print("alpha: ", alpha)
        self.grid_vals = (1 - alpha) * self.target_grid_vals + alpha * vals
