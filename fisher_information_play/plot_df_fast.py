import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import time

start_time = time.time()

num_pts = 100

mean = np.load('belief_mean_snapshot_t200.npy')
cov = np.load('belief_cov_snapshot_t200.npy')

N = int((mean.shape[0]-1) / 2)
print("size N: ", N)

fig = plt.figure()

# parse data
robot_mean = mean[0:3]
robot_cov = cov[0:3, 0:3]

landmark_mean = []
landmark_cov = []
lm_det = []
for i in range(N-1):
    temp_mean = mean[2+2*i+1 : 2+2*i+3]
    temp_cov = cov[2+2*i+1 : 2+2*i+3, 2+2*i+1 : 2+2*i+3]
    if temp_mean[0]!=0 or temp_mean[1]!=0:
        landmark_mean.append(temp_mean)
        landmark_cov.append(temp_cov)
        lm_det.append(np.linalg.det(temp_cov))
    else:
        pass
landmark_mean = np.array(landmark_mean)
landmark_cov = np.array(landmark_cov)
lm_det = np.array(lm_det)

# plot robot and landmarks
ax1 = fig.add_subplot(121)
ax1.set_aspect('equal')
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)

ax1.scatter(robot_mean[0], robot_mean[1], color='red')
for i in range(landmark_mean.shape[0]):
    ax1.scatter(landmark_mean[i][0], landmark_mean[i][1], color='blue')
    # ax1.annotate('{:.2E}'.format(np.linalg.det(landmark_cov[i])), landmark_mean[i][0:2])
    ax1.annotate('{}'.format(i), landmark_mean[i][0:2])

# compute distance field
temp_grid = np.meshgrid(*[np.linspace(0, 20, num_pts) for _ in range(2)])
grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
# fast implementation of uncertainty distance field
grid_x = grid[:, 0]
grid_y = grid[:, 1]
state_x = landmark_mean[:, 0][:, np.newaxis]
state_y = landmark_mean[:, 1][:, np.newaxis]
diff_x = grid_x - state_x
diff_y = grid_y - state_y
dist_xy = np.sqrt(diff_x**2 + diff_y**2)
dist_xy = np.exp(-dist_xy).T * ( -np.log(lm_det) )
vals = np.sum(dist_xy, axis=1)

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

vals /= np.sum(vals)
ax2.contourf(*xy, vals.reshape(num_pts,num_pts), levels=25)

'''
for i in range(landmark_mean.shape[0]):
    ax2.scatter(landmark_mean[i][0], landmark_mean[i][1], color='white', s=10)
    ax2.annotate('{}'.format(i), landmark_mean[i])
'''

plt.show()
