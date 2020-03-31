import autograd.numpy as np
from math import pi, sqrt
from matplotlib import pyplot as plt


means = np.load("/home/msun/Code/ErgodicBSP/test/mean_mat.npy")
raw_vars = np.load("/home/msun/Code/ErgodicBSP/test/cov_mat.npy")

def single_gaussian(x, mean, var):
    p = 1 / np.sqrt(2 * pi * var) * np.exp(-0.5 * (x - mean)**2 / var)
    return p

def multi_gaussian(x, mean, var):
    p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
        -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
    return p

num_pts = 50
size = 20
temp_grid = np.meshgrid(*[np.linspace(0, size, num_pts) for _ in range(2)])
grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
print(grid.shape)

nStates = 3
nLandmark = int((means.shape[0] - nStates) / 2)
observed_table = np.zeros(nLandmark)
vars = []
for i in range(nLandmark):
    var = raw_vars[2 + 2 * i + 1: 2 + 2 * i + 3, 2 + 2 * i + 1: 2 + 2 * i + 3]
    if var[0,0] == 99999999.:
        observed_table[i] = 0
    else:
        observed_table[i] = 1
    vars.append(var)
    print(var)
# vars = np.array(vars)

vals = np.zeros(grid.shape[0])
for i in range(grid.shape[0]):
    for j in range(nLandmark):
        if observed_table[j] == 1:
            x = grid[i, :]
            mean = means[2 + 2 * j + 1: 2 + 2 * j + 3]
            print("mean: ", mean)
            var = vars[j]
            p = multi_gaussian(x, mean, var)
            vals[i] += p
        else:
            pass
vals = np.array(vals)
vals /= np.sum(vals)

xy = []
for g in grid.T:
    xy.append(
        np.reshape(g, newshape=(num_pts, num_pts))
    )
grid_vals = vals.reshape(num_pts, num_pts)

for i in range(5):
    for j in range(5):
        print(grid_vals[20 + i, 20 + j])

fig = plt.figure()
ax = fig.gca()
ax.set_aspect('equal', 'box')
ax.contourf(*xy, grid_vals, levels=20)
plt.show()
