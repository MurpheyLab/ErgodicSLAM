import numpy as np
import matplotlib.pyplot as plt


xt_est = np.load('xt_est.npy')
xt_true = np.load('xt_true.npy')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

ax.scatter(xt_true[:,0], xt_true[:,1], s=1)
ax.scatter(xt_est[:,0], xt_est[:,1], s=10)

plt.grid()
plt.show()

