import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal as mvn


xy = np.meshgrid(*[np.linspace(0,20,51) for _ in range(2)])
vals = np.zeros((50,50))

grids = np.array([xy[0][0:50, 0:50].ravel(), xy[1][0:50, 0:50].ravel()]).T
print(grids.shape)
rv = mvn([10.,10.], np.array([[1.0,0.5],[0.5,1.0]]))
grid_vals = rv.pdf(grids).reshape(50,50)

fig = plt.figure()
ax = fig.add_subplot(111)

levels = MaxNLocator(nbins=125).tick_values(grid_vals.min(), grid_vals.max())
cmap = plt.get_cmap('hot')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
p = ax.pcolormesh(xy[0], xy[1], grid_vals, cmap=cmap, norm=norm)
ax.set_aspect('equal')
# ax.set_xticks(xy[0][0])
# ax.set_yticks(xy[1][:,0])
plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
plt.show()
