import numpy as np
from numpy import exp, sqrt
import matplotlib.pyplot as plt

# landmarks = np.load('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_7.npy')
landmarks = np.load('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/center_single.npy')

num_pts = 50
size = 15

temp_grid = np.meshgrid(*[np.linspace(0, size, num_pts) for _ in range(2)])
grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
grid_vals = np.zeros(grid.shape[0])

print(grid.shape)

'''
for lm in landmarks:
    mat11 = (grid[:,0]-lm[0]) * sqrt( (grid[:,0]-lm[0])**2 + (grid[:,1]-lm[1])**2 )
    mat21 = (grid[:,1]-lm[1]) * sqrt( (grid[:,0]-lm[0])**2 + (grid[:,1]-lm[1])**2 )
    mat12 = 2*(grid[:,0]-lm[0])*(grid[:,1]-lm[1]) / ( (grid[:,0]-lm[0])**2 + (grid[:,1]-lm[1])**2 )
    mat22 = -2*(grid[:,0]-lm[0])*(grid[:,1]-lm[1]) / ( (grid[:,0]-lm[0])**2 + (grid[:,1]-lm[1])**2 )
    grid_vals += mat11 * mat22 - mat12 * mat21
'''

def fisher_mat(r, lm, cov_inv):
    dist = sqrt( (r[0]-lm[0])**2 + (r[1]-lm[1])**2 )
    if dist > 40:
        return 0
    else:
        dm11 = (r[0]-lm[0]) / dist
        dm12 = (r[1]-lm[1]) / dist
        # dm21 = 2*(r[0]-lm[0])*(r[1]-lm[1]) / (dist**2)
        # dm22 =-2*(r[0]-lm[0])*(r[1]-lm[1]) / (dist**2)
        # dm21 = 1 / ( 1 + (lm[1]-r[1])**2/(lm[0]-r[0])**2 ) * ( (lm[1]-r[1])/(lm[0]-r[0])**2 )
        dm21 = (lm[1]-r[1]) / (dist**2)
        # dm22 = 1 / ( 1 + (lm[1]-r[1])**2/(lm[0]-r[0])**2 ) * ( -1/(lm[0]-r[0]) )
        dm22 = -(lm[0]-r[0]) / (dist**2)
        dm = np.array([[dm11,dm12],[dm21,dm22]])
        # fim = np.dot(np.dot(dm, cov_inv), dm.T)
        fim = np.dot(np.dot(dm.T, cov_inv), dm)
        return np.linalg.det(fim)
        # return np.trace(fim)

cov = np.array([[0.01, 0],[0, 0.01]])
cov_inv = np.linalg.inv(cov)

for i in range(grid.shape[0]):
    r = grid[i]
    for lm in landmarks:
        grid_vals[i] += fisher_mat(r, lm, cov_inv)

xy = []
for g in grid.T:
    xy.append(
        np.reshape(g, newshape=(num_pts, num_pts))
    )
vals = grid_vals.reshape(num_pts, num_pts)

plt.contourf(*xy, vals, levels=20)
# plt.scatter(landmarks[:, 0], landmarks[:, 1], color='white', marker='P')
ax = plt.gca()
ax.set_aspect('equal', 'box')

plt.show()
