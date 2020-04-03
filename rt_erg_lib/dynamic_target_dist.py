# import rospy

import numpy as np
import numpy.random as npr
from math import pi


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

    def update1(self, nStates, nLandmark, observed_table, belief_means, belief_vars):
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
            vals /= np.sum(vals)

            print("replanning")
            alpha = p / (p + threshold)
            # self.grid_vals = (1 - alpha) * self.grid_vals + alpha * vals
            self.grid_vals = vals

    def update2(self, nStates, nLandmark, observed_table, belief_means, belief_vars):
        p = np.linalg.det(belief_vars[0: nStates, 0: nStates])
        print("\np: ", p)

        temp_grid = np.meshgrid(*[np.linspace(0, self.size, self.num_pts) for _ in range(2)])
        grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

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
        if np.sum(vals) != 0:
            vals /= np.sum(vals)

        # threshold = 1e-03
        threshold = 4e-04
        alpha = (p / (p + threshold)) ** 2
        print("alpha: ", alpha)
        self.grid_vals = (1 - alpha) * self.target_grid_vals + alpha * vals
