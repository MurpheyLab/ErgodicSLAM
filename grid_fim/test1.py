import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal as mvn


npts = 50
x, y = np.meshgrid(*[np.linspace(0, 20, npts+1) for _ in range(2)])
grid = np.c_[x[0:npts,0:npts].ravel(), x[0:npts,0:npts].ravel()]
dx = grid[1][0] - grid[0][0]
xax = x[0]
yax = y[:,0]
vals = np.zeros(npts**2)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

cmap = plt.get_cmap('hot')
levels = MaxNLocator(nbins=50).tick_values(0., 1.)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
p = ax.pcolormesh(x, y, vals.reshape(npts, npts), cmap=cmap, norm=norm, edgecolors='k', linewidth=0.004)

plt.show()

