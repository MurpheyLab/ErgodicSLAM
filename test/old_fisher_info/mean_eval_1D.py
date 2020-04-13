import numpy as np
from math import pi, e


def single_gaussian(x, mean, var):
    p = 1 / np.sqrt(2 * pi * var) * e ** (-0.5 * (x - mean)**2 / var)
    return p
print('gaussian test: ', single_gaussian(-3, 0, 1))


n_pts = 1000000
data = np.random.randn(n_pts)
# print('data: ', data)
val = 0
vals = []
for i in range(n_pts):
    x = data[i]
    px = single_gaussian(x, 0, 1)
    # print('x * px: ', x * px)
    val += x * px
    vals.append(x * px)
print(val)
print(np.var(data))
print(np.var(vals))
