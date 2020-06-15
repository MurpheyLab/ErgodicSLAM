'''
first working fim implementation
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm

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

# compute FIM field
mcov = np.diag([0.0225, 0.01]) ** 2
imcov = np.linalg.inv(mcov)
temp_grid = np.meshgrid(*[np.linspace(0, 20, num_pts) for _ in range(2)])
grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
vals = np.zeros(grid.shape[0])

# set robot position as target (assume known landmark locations)
# landmark_mean = np.array([[10., 10.]])

robot_mean = np.array([10., 10., 0.])
'''
for i in range(grid.shape[0]):
    # assume robot position (target parameter) is here
    r = grid[i]
    for j in range(landmark_mean.shape[0]):
        l = landmark_mean[j]
        d = np.sqrt((r[0]-l[0])**2 + (r[1]-l[1])**2)
        if d <= 4:
            dm = np.array([[(r[0]-l[0])/d, (r[1]-l[1])/d],
                           [-(r[1]-l[1])/d**2, (r[0]-l[0])/d**2]])
            fim = dm.T @ imcov @ dm
            factor = mvn.pdf(r, robot_mean[0:2], robot_cov[0:2,0:2])
            vals[i] += np.linalg.det(fim) * factor
        else:
            pass
'''

'''
# set landmark position as target (assume known robot location)
for i in range(landmark_mean.shape[0]):
    # analyze i-th landmark
    for j range(grid.shape[0]):
        # if it's at this location
        l = grid[j]
        for k in range(grid.shape[0]):
            # then for any grid in the space
            # we have a fim
            r = grid[k]
'''
vals = np.zeros(grid.shape[0])
for k in range(landmark_mean.shape[0]):
    print("k: [{}/{}] --> pos: [{}, {}]".format(k, landmark_mean.shape[0], landmark_mean[k][0], landmark_mean[k][1]))
    temp_vals = np.zeros(grid.shape[0])
    for i in tqdm(range(grid.shape[0])):
        l = grid[i]
        for j in range(grid.shape[0]):
            r = grid[j]
            d = np.sqrt((r[0]-l[0])**2 + (r[1]-l[1])**2)
            if d <= 4 and d > 1e-06:
                dm = np.array([[-(r[0]-l[0])/d, -(r[1]-l[1])/d],
                               [(r[1]-l[1])/d**2, -(r[0]-l[0])/d**2]])
                fim = dm.T @ imcov @ dm
                temp_vals[j] += np.linalg.det(fim) * mvn.pdf(l, landmark_mean[k], landmark_cov[k])
                # temp_vals[j] += np.linalg.det(fim) * mvn.pdf(l, landmark_mean[k], np.diag([1e-01, 5e-02])) # for debug only
            else:
                pass
    print("[{}] --> sum(temp_vals): {}".format(k, np.sum(temp_vals)))
    vals += temp_vals

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
