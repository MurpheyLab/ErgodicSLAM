'''
full expectation fim implementation
(fast version)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm
import time


num_pts = 50

mean = np.load('belief_mean_snapshot_t50.npy')
cov = np.load('belief_cov_snapshot_t50.npy')

N = int((mean.shape[0]-1) / 2)
print("size N: ", N)

fig = plt.figure()

# parse data
robot_mean = mean[0:3]
robot_cov = cov[0:3, 0:3]

landmark_mean = []
landmark_cov = []
for i in range(N-1):
    temp_mean = mean[2+2*i+1 : 2+2*i+3]
    temp_cov = cov[2+2*i+1 : 2+2*i+3, 2+2*i+1 : 2+2*i+3]
    if temp_mean[0]!=0 or temp_mean[1]!=0:
        landmark_mean.append(temp_mean)
        landmark_cov.append(temp_cov)
    else:
        pass
landmark_mean = np.array(landmark_mean)
landmark_cov = np.array(landmark_cov)

# debug only:
# landmark_mean = np.array([[5., 3.],[5., 5.5]])

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


start_time = time.time()
# compute FIM field
mcov = np.diag([0.0225, 0.01]) ** 2
imcov = np.linalg.inv(mcov)
temp_grid = np.meshgrid(*[np.linspace(0, 20, num_pts) for _ in range(2)])
grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]

vals = np.zeros(grid.shape[0])

grid_x = grid[:,0]
grid_y = grid[:,1]
diff_x = grid_x - grid_x[:, np.newaxis]
diff_y = grid_y - grid_y[:, np.newaxis]
dist_xy = np.sqrt(diff_x**2 + diff_y**2)

range_flag = dist_xy < 40
range_flag = range_flag.astype(int)

zero_flag = np.ones((num_pts**2, num_pts**2)) - np.eye(num_pts**2)

dist_xy = dist_xy + 1e-09

lm_x = grid[:,0]
lm_y = grid[:,1]

dm11 = -(grid_x - lm_x[:, np.newaxis]) / dist_xy * zero_flag
dm12 = -(grid_y - lm_y[:, np.newaxis]) / dist_xy * zero_flag
dm21 = (grid_y - lm_y[:, np.newaxis]) / dist_xy**2 * zero_flag
dm22 = -(grid_x - lm_x[:, np.newaxis]) / dist_xy**2 * zero_flag

fim11 = dm11 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm21 * (dm11*imcov[0,1] + dm21*imcov[1,1])
fim12 = dm12 * (dm11*imcov[0,0] + dm21*imcov[1,0]) + dm22 * (dm11*imcov[0,1] + dm21*imcov[1,1])
fim21 = dm11 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm21 * (dm12*imcov[0,1] + dm22*imcov[1,1])
fim22 = dm12 * (dm12*imcov[0,0] + dm22*imcov[1,0]) + dm22 * (dm12*imcov[0,1] + dm22*imcov[1,1])

det = fim11 * fim22 - fim12 * fim21
det = det * range_flag

# for i in range(landmark_mean.shape[0]):
for i in range(1):
    distr = mvn.pdf(grid, landmark_mean[i], landmark_cov[i])
    det = det * distr[:, np.newaxis]
    det_vals = np.sum(det, axis=0)
    vals += det_vals

print("elapsed time: ", time.time() - start_time)

# plot fim field
xy = []
for g in grid.T:
    xy.append(
        np.reshape(g, newshape=(num_pts, num_pts))
    )

ax2 = fig.add_subplot(122)
ax2.set_aspect('equal')
ax2.set_xlim(0, 20)
ax2.set_ylim(0, 20)

print('sum(vals): ', np.sum(vals))
if np.sum(vals) != 0:
    vals /= np.sum(vals)

ax2.contourf(*xy, vals.reshape(num_pts,num_pts), levels=25)

'''
for i in range(landmark_mean.shape[0]):
    ax2.scatter(landmark_mean[i][0], landmark_mean[i][1], color='white', s=10)
    ax2.annotate('{}'.format(i), landmark_mean[i])
'''

plt.show()
