'''
full expectation fim implementation
(fast version)
plot with grid instead of contourf
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm
import time


num_pts = 80

mean = np.load('belief_mean_snapshot_t100.npy')
cov = np.load('belief_cov_snapshot_t100.npy')

N = int((mean.shape[0]-1) / 2)
print("size N: ", N)

fig = plt.figure()

# parse data
robot_mean = mean[0:3]
robot_cov = cov[0:3, 0:3]

temp_grid = np.meshgrid(*[np.linspace(0, 20, num_pts+1) for _ in range(2)])
grid = np.c_[temp_grid[0][0:num_pts,0:num_pts].ravel(), temp_grid[1][0:num_pts,0:num_pts].ravel()]
dx = grid[1][0] - grid[0][0]
print('dx: ', dx)
xaxis = temp_grid[0][0]
yaxis = temp_grid[0][0]

landmark_mean = []
clip_landmark_mean = []
landmark_cov = []
for i in range(N-1):
    temp_mean = mean[2+2*i+1 : 2+2*i+3]
    temp_cov = cov[2+2*i+1 : 2+2*i+3, 2+2*i+1 : 2+2*i+3]
    if temp_mean[0]!=0 or temp_mean[1]!=0:
        # landmark_mean.append(temp_mean)
        landmark_cov.append(temp_cov)
        print('temp_mean: ', temp_mean)
        clip_mean = np.array([xaxis[int(temp_mean[0]/dx)], yaxis[int(temp_mean[1]/dx)]])
        print('clip_mean: ', clip_mean)
        landmark_mean.append(clip_mean)
    else:
        pass
landmark_mean = np.array(landmark_mean)
landmark_cov = np.array(landmark_cov)
print('landmark_mean:\n', landmark_mean)

# debug only:
# landmark_mean = np.array([[5., 3.],[5., 5.5]])

start_time = time.time()
# compute FIM field
mcov = np.diag([0.0225, 0.01]) ** 2
# mcov = np.diag([1, 1]) ** 1
imcov = np.linalg.inv(mcov)
# imcov = np.eye(2)

'''
"""for debug only: make some data"""
landmark_mean = np.array([[4., 4.],
                           [16., 16.]])
landmark_cov = np.array([np.array([[0.0001, 0.00002], [0.00002, 0.0001]]),
                         np.array([[0.002, 0.0001], [0.0001, 0.002]])]) * 1
'''

# start computation
vals = np.zeros(grid.shape[0])

grid_x = grid[:,0]
grid_y = grid[:,1]
diff_x = grid_x - grid_x[:, np.newaxis]
diff_y = grid_y - grid_y[:, np.newaxis]
dist_xy = np.sqrt(diff_x**2 + diff_y**2)

range_flag_1 = 0.0 < dist_xy
range_flag_1 = range_flag_1.astype(int)
range_flag_2 = dist_xy < 4.0 # setup range limit here
range_flag_2 = range_flag_2.astype(int)

range_flag = range_flag_1 * range_flag_2

zero_flag = np.ones((num_pts**2, num_pts**2)) - np.eye(num_pts**2)

dist_xy = dist_xy + 1e-09

dm11 = -(grid_x - grid_x[:, np.newaxis]) / dist_xy * zero_flag # grid_* one for robot, one for landmark
dm12 = -(grid_y - grid_y[:, np.newaxis]) / dist_xy * zero_flag
dm21 = (grid_y - grid_y[:, np.newaxis]) / dist_xy**2 * zero_flag
dm22 = -(grid_x - grid_x[:, np.newaxis]) / dist_xy**2 * zero_flag

fim11 = dm11 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm21 * (dm11*imcov[0,1] + dm21*imcov[1,1])
fim12 = dm12 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm22 * (dm11*imcov[0,1] + dm21*imcov[1,1])
fim21 = dm11 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm21 * (dm12*imcov[0,1] + dm22*imcov[1,1])
fim22 = dm12 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm22 * (dm12*imcov[0,1] + dm22*imcov[1,1])

det = fim11 * fim22 - fim12 * fim21
# det = fim11 + fim22
det = det * range_flag


# when we know priori of landmarks
for i in range(landmark_mean.shape[0]):
# for i in [7]:
    distr = mvn.pdf(grid, landmark_mean[i], landmark_cov[i] * 1)
    # distr = np.log10(distr + 1.)
    scaled_det = det * distr[:, np.newaxis]
    det_vals = np.sum(scaled_det, axis=0)
    print("det_vals: ", np.sum(det_vals))
    vals += det_vals


# when we don't
# vals += np.sum(det, axis=0)

print("elapsed time: ", time.time() - start_time)


###############################################
# plot

# plot robot and landmarks
ax1 = fig.add_subplot(121)
ax1.set_aspect('equal')
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)

ax1.scatter(robot_mean[0], robot_mean[1], color='red')
for i in range(landmark_mean.shape[0]):
    ax1.scatter(landmark_mean[i][0], landmark_mean[i][1], color='blue')
    ax1.annotate('{:.2E}'.format(np.linalg.det(landmark_cov[i])), landmark_mean[i][0:2])
    print("[{}]: {:.2E}".format(i, np.linalg.det(landmark_cov[i])))


# plot fim field
xy = []
for g in grid.T:
    xy.append(
        np.reshape(g, newshape=(num_pts, num_pts))
    )

ax2 = fig.add_subplot(122)
ax2.set_aspect('equal')
# ax2.set_xlim(0, 20)
# ax2.set_ylim(0, 20)

print('sum(vals): ', np.sum(vals))
if np.sum(vals) != 0:
    vals /= np.sum(vals)
vals = vals.reshape(num_pts, num_pts)

levels = MaxNLocator(nbins=125).tick_values(vals.min(), vals.max())
cmap = plt.get_cmap('hot')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# ax2.contourf(*xy, vals.reshape(num_pts,num_pts), levels=50, cmap=cm.hot)
p = ax2.pcolormesh(temp_grid[0], temp_grid[1], vals, cmap=cmap, norm=norm)

'''
for i in range(landmark_mean.shape[0]):
    ax2.scatter(landmark_mean[i][0], landmark_mean[i][1], color='white', s=10)
    ax2.annotate('{}'.format(i), landmark_mean[i])
'''

plt.show()
