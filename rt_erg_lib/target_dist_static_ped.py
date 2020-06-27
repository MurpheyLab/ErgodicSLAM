# import rospy

import numpy as np
import numpy.random as npr

class TargetDist(object):
    '''
    This is going to be a test template for the code,
    eventually a newer version will be made to interface with the
    unity env
    '''

    # 2020-03-01: add "size" parameter to support customizable exploration area size
    def __init__(self, num_pts, ped_state, bounds, size):

        # TODO: create a message class for this
        # rospy.Subscriber('/target_distribution',  CLASSNAME, self.callback)

        self.num_pts = num_pts
        grid = np.meshgrid(*[np.linspace(0, size, num_pts), np.linspace(0, size, num_pts)])
        self.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        self.ped_state = ped_state
        self.bounds = bounds

        self.has_update = False
        self.grid_vals = self.__call__()

    def get_grid_spec(self):
        xy = []
        for g in self.grid.T:
            xy.append(
                np.reshape(g, newshape=(self.num_pts, self.num_pts))
            )
        return xy, self.grid_vals.reshape(self.num_pts, self.num_pts)


    def __call__(self, threshold=1.0):
        grid_x = self.grid[:,0]
        grid_y = self.grid[:,1]

        space = np.array(self.bounds).reshape(-1,2)
        space_x = space[:,0]
        space_y = space[:,1]

        state = self.ped_state
        state_x = np.concatenate((state[:,0], space_x))[:, np.newaxis]
        state_y = np.concatenate((state[:,1], space_y))[:, np.newaxis]

        diff_x = grid_x - state_x
        diff_y = grid_y - state_y
        dist_xy = np.sqrt(diff_x**2 + diff_y**2)

        dist_flag = dist_xy > threshold
        dist_flag = dist_flag.astype(int)
        dist_xy *= dist_flag

        dist_val = dist_xy.min(axis=0)
        dist_val /= np.sum(dist_val)
        return dist_val
