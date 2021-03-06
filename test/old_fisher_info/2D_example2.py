import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import pi


def multi_gaussian(x, mean, var):
    p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
        -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
    return p

if __name__ == '__main__':
    ######
    # parameter initialization
    n_points = 1000
    size = (50, 50)
    sig = 0.1
    points = np.random.randn(2, n_points) * sig
    mean = [np.array([0, 0])]
    points[0,:] += mean[0][0]
    points[1,:] += mean[0][1]
    fig = plt.figure()

    # calculate covariance matrix of the randomly generated points
    cov = np.cov(points)
    print('cov:\n', end='')
    print(cov, end='\n\n')

    ######
    # first figure: point data sample
    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.scatter(points[0,:], points[1,:], s=0.2)

    ######
    # calculate fisher information
    temp_grid = np.meshgrid(*[np.linspace(-4, 4, size[0]) for _ in range(2)])
    grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
    vals = np.zeros(grid.shape[0])
    # for each grid 'mu' in search space, calculate fisher information
    # from all the points 'x'
    # for i in range(grid.shape[0]):
    for i in tqdm(range(grid.shape[0])):
        fish_mat = 0
        # for j in range(n_points):
        for j in range(1):
            mu = grid[i, :]
            # x = points[:, j]
            x = mean[j]
            diff = x - mu
            ## method 1
            # fish_mat = np.outer(diff, diff)
            # vals[i] += np.linalg.det(fish_mat) * multi_gaussian(x, mu, np.eye(2))
            ## method 2
            cov = np.array([[sig**2, 0], [0, sig**2]])
            inner_mat = np.outer(diff, diff)
            inner_mat = np.dot(cov, inner_mat)
            inner_mat = np.dot(inner_mat, np.linalg.inv(cov).T)
            fish_mat += inner_mat * multi_gaussian(x, mu, cov)
            ## method 3
            # fish_mat += np.linalg.inv(cov) * multi_gaussian(x, mu, cov)
        vals[i] = np.linalg.det(fish_mat)
    # reshape grid and vals for visualization
    xy = []
    for g in grid.T:
        xy.append(
            np.reshape(g, newshape=size)
        )
    vals = vals / np.sum(vals)
    vals = vals.reshape(size[0], size[1])

    ######
    # second figure: visualize Fisher information
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.contourf(*xy, vals, levels=20)

    plt.show()
