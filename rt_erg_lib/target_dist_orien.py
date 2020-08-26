# import rospy

import numpy as np
import numpy.random as npr
from numpy import sin, cos, pi
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
        grid = np.meshgrid(np.linspace(0, size, num_pts), np.linspace(0, size, num_pts), np.linspace(-pi, pi, num_pts))
        # self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]
        self.grid = np.stack([g.ravel() for g in grid]).T

        # self.means = [npr.uniform(0.2, 0.8, size=(2,))
        #                     for _ in range(num_nodes)]
        # self.vars  = [npr.uniform(0.05, 0.2, size=(2,))**2
        #                     for _ in range(num_nodes)]
        self.means = means
        self.vars  = vars

        # print("means: ", self.means)

        self.has_update = False
        self.grid_vals = self.__call__(self.grid)

    def get_grid_spec(self):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(self.num_pts, self.num_pts))
            )
        return xy, self.grid_vals.reshape(self.num_pts, self.num_pts)


    def __call__(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 3, 'Does not have right exploration dim'

        val = np.zeros(x.shape[0])

        '''
        for m, v in zip(self.means, self.vars):
            innerds = np.sum((x-m)**2 / v, 1)
            val += np.exp(-innerds/2.0) / np.sqrt((2*np.pi)**2 * np.prod(v))
        # normalizes the distribution
        val /= np.sum(val)
        '''
        for m, v in zip(self.means, self.vars):
            rv = mvn(mean=m, cov=np.diag(v))
            val += rv.pdf(x)

        return val
