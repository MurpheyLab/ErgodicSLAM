# import rospy

import numpy as np
import numpy.random as npr
from numpy import pi, sqrt, exp
from .utils import *
from scipy.stats import multivariate_normal


class TargetDist(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    # 2020-03-01: add "size" parameter to support customizable exploration area size
    def __init__(self, num_pts, means, vars, size, landmarks, mcov, sensor_range):

        # TODO: create a message class for this
        # rospy.Subscriber('/target_distribution',  CLASSNAME, self.callback)

        self.num_pts = num_pts
        self.size = size
        grid = np.meshgrid(*[np.linspace(0, size, num_pts) for _ in range(2)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]
        self.landmarks = landmarks
        self.cov_inv = np.linalg.inv(mcov)
        self.sensor_range = sensor_range
        self.switch_counter = 0 # comeback

        # self.means = [npr.uniform(0.2, 0.8, size=(2,))
        #                     for _ in range(num_nodes)]
        # self.vars  = [npr.uniform(0.05, 0.2, size=(2,))**2
        #                     for _ in range(num_nodes)]
        self.means = means
        self.vars = vars

        # print("means: ", self.means)

        self.has_update = False
        self.grid_vals = self.__call__(self.grid) # the actual grid vals that robot set as target
        self.target_grid_vals = self.__call__(self.grid) # target grid vals set as task
        self.belief_vals = self.init_fim()

    def init_fim(self): # init fim (just once)
        vals = np.zeros(self.grid.shape[0])
        for i in range(self.grid.shape[0]):
            r = self.grid[i]
            for lm in self.landmarks:
                # compute fim
                fim = 0
                dist = sqrt( (r[0]-lm[0])**2 + (r[1]-lm[1])**2 )
                if(dist <= self.sensor_range):
                    dm11 = (r[0]-lm[0]) * dist
                    dm12 = (r[1]-lm[1]) * dist
                    dm21 = (lm[1]-r[1]) / (dist**2)
                    dm22 =-(lm[0]-r[0]) / (dist**2)
                    dm = np.array([[dm11, dm12], [dm21, dm22]])
                    fim = np.linalg.det( np.dot(np.dot(dm.T, self.cov_inv), dm) )
                vals[i] += fim
        if(np.sum(vals) != 0): # normalize
            vals /= np.sum(vals)
        return vals


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

    def update0(self):
        self.grid_vals = self.belief_vals

    def update1(self, nStates, belief_means, belief_cov, threshold=1e-3):
        '''
        hard switch with FIM field, but with sticking mechanism
        '''
        p = np.linalg.det(belief_cov[0: nStates, 0: nStates])
        print("p: ", p)
        if p < threshold:
            if self.switch_counter == 0:
                self.grid_vals = self.target_grid_vals
            else:
                if self.switch_counter < 50:
                    self.switch_counter += 1
                    print(self.switch_counter)
                    self.grid_vals = self.belief_vals
                else:
                    self.grid_vals = self.target_grid_vals # replace with "hard" switch
                    self.switch_counter = 0
        else:
            self.grid_vals = self.belief_vals
            if self.switch_counter == 0:
                self.switch_counter = 1


    def update2(self, nStates, belief_means, belief_cov, threshold=1e-3):
        '''
        trivial hard switch with FIM field
        '''
        p = np.linalg.det(belief_cov[0: nStates, 0: nStates])
        # print("\np: ", p)
        if p < threshold:
            self.grid_vals = self.target_grid_vals
        else:
            self.grid_vals = self.belief_vals


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
