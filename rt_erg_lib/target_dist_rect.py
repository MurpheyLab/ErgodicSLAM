# import rospy

import numpy as np
import numpy.random as npr
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
        self.size = size
        self.num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, size[0], num_pts),
                             np.linspace(0, 2., 20)])
        self.xy = np.array([grid[0], grid[1]])
        print("origin grid: ", grid[1])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]
        print("self.grid.shape: ", self.grid)
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
            print("g: ", g)
            xy.append(
                np.reshape(g, newshape=(self.num_pts, int(self.num_pts*self.size[1]/self.size[0])))
            )
        xy = np.array(xy)
        return self.xy, self.grid_vals.reshape(20, self.num_pts)


    def __call__(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        assert x.shape[1] == 2, 'Does not have right exploration dim'

        val = np.zeros(x.shape[0])
        for m, v in zip(self.means, self.vars):
            # innerds = np.sum((x-m)**2 / v, 1)
            # val += np.exp(-innerds/2.0) / np.sqrt((2*np.pi)**2 * np.prod(v))
            val += mvn.pdf(x, m, np.diag(v))
        # normalizes the distribution
        val /= np.sum(val)
        print("val: ", val)
        return val
