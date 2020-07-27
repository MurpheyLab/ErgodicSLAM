import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal as mvn
import time


size = 20.
num_pts = 50
sensor_range = 4.
raw_grid = np.meshgrid(*[np.linspace(0, size, num_pts+1) for _ in range(2)])
grid = [raw_grid[0][0:num_pts, 0:num_pts], raw_grid[1][0:num_pts, 0:num_pts]]
grid2 = np.c_[grid[0].ravel(), grid[1].ravel()]
grid_x = grid2[:,0]
grid_y = grid2[:,1]

mean = np.array([6., 12.])
cov = np.eye(2) * 1e-03 * 5
rv = mvn(mean=mean, cov=cov)
prob = 0.5 * np.exp( -1. / (rv.pdf(grid2)+1e-09) ) ** 5
print('det cov: ', np.linalg.det(cov))
print('prob max: ', np.max(prob))

start_1 = time.time()
vals = np.ones(num_pts**2) * 0.5
for i in range(num_pts**2):
    cell = grid2[i]
    dist_x = (grid_x-cell[0])**2 + (grid_y-cell[1])**2
    dist_flag = (dist_x < sensor_range**2).astype(int)
    addi = prob * dist_flag
    vals[i] += np.sum(addi)
vals = np.clip(vals, a_min=None, a_max=1.)
vals = vals.reshape(num_pts, num_pts)
print('time_1: ', time.time()-start_1)

start_2 = time.time()
vals = np.ones(num_pts**2) * 0.5
dist_xy = (grid_x-grid_x[:,np.newaxis])**2 + (grid_y-grid_y[:,np.newaxis])**2
dist_flag = (dist_xy < sensor_range**2).astype(int)
addi = np.sum(dist_flag * prob[:,np.newaxis], axis=0)
vals += addi
vals = np.clip(vals, a_min=None, a_max=1.)
vals = vals.reshape(num_pts, num_pts)
print('time_2: ', time.time()-start_2)

fig = plt.figure()
ax = fig.add_subplot(111)
cmap = plt.get_cmap('gray')
levels = MaxNLocator(nbins=25).tick_values(vals.min(), vals.max())
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
p = ax.pcolormesh(grid[0], grid[1], vals, cmap=cmap, edgecolors='k', linewidth=0.004, norm=norm)
ax.set_aspect('equal')
plt.colorbar(p, fraction=0.046, pad=0.04, ax=ax)
plt.show()

