import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

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

# plot robot and landmarks
ax1 = fig.add_subplot(121)
ax1.set_aspect('equal')
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)

ax1.scatter(robot_mean[0], robot_mean[1], color='red')
for i in range(landmark_mean.shape[0]):
    ax1.scatter(landmark_mean[i][0], landmark_mean[i][1], color='blue')

# compute FIM field
mcov = np.diag([0.0225, 0.01]) ** 2 * 100000
imcov = np.linalg.inv(mcov)
temp_grid = np.meshgrid(*[np.linspace(0, 20, num_pts) for _ in range(2)])
grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
vals = np.zeros(grid.shape[0])
for i in range(grid.shape[0]):
    r = grid[i]
    # for j in range(landmark_mean.shape[0]):
    for j in [4]:
        l = landmark_mean[j]
        cov = landmark_cov[j]
        dist = np.sqrt((r[0]-l[0])**2 + (r[1]-l[1])**2)
        if(dist > 4):
            continue
        '''
        temp_mat = np.array([[l[0]-r[0], l[1]-r[1], r[0]-l[0], r[1]-l[1]],
                             [r[1]-l[1], l[0]-r[0], l[1]-r[1], r[0]-l[0]]])
        '''

        temp_mat = np.array([[l[0]-r[0], l[1]-r[1]],
                             [r[1]-l[1], l[0]-r[0]]])

        # temp_mat[0,:] *= 2
        # temp_mat[1,:] /= dist**2
        temp_mcov = mcov * dist**0.5
        imcov = np.linalg.inv(temp_mcov)
        fim = temp_mat.T @ imcov @ temp_mat
        vals[i] += np.linalg.det(fim) * (mvn.pdf(l, l, cov)) #+ mvn.pdf(r, r, robot_cov[0:2, 0:2]))

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
