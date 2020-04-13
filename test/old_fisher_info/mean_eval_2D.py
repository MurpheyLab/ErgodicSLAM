import numpy as np
from math import pi, e
import matplotlib.pyplot as plt


def multi_gaussian(x, mean, var):
    p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
        -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
    return p
print('gaussian test: ', multi_gaussian(np.array([0, 0]),
                                         np.array([0, 0]),
                                         np.eye(2)))


n_pts = 100000
data = np.random.randn(n_pts, 2) + 1
# print('data: ', data)
val = 0
vals = []
for i in range(n_pts):
    x = data[i]
    px = multi_gaussian(x, np.array([0, 0]), np.eye(2))
    # print('x: {} , px: {}'.format(x, px))
    val += x * px
    vals.append(x * px)
vals = np.array(vals)
print(val)
print(np.cov(vals.T))

fig = plt.figure()
ax = fig.gca()
ax.scatter(vals[:,0], vals[:,1], s=0.2)
ax.set_aspect('equal', 'box')
plt.show()

