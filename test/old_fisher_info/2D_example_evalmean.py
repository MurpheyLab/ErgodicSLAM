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
    points = np.random.randn(2, n_points)
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
    temp_grid = np.meshgrid(*[np.linspace(-4, 4, 50) for _ in range(2)])
    grid = np.c_[temp_grid[0].ravel(), temp_grid[1].ravel()]
    vals = np.zeros(grid.shape[0])
    # for each grid 'mu' in search space, calculate fisher information
    # from all the points 'x'
    # for i in range(grid.shape[0]):
    for i in tqdm(range(grid.shape[0])):
        mu = grid[i, :]
        x = np.array([0, 0])
        diff = x - mu
        inner_mat = np.outer(diff, diff)
        inner_mat = np.dot(cov, inner_mat)
        inner_mat = np.dot(inner_mat, np.linalg.inv(cov).T)
        fish_mat = inner_mat * multi_gaussian(x, mu, np.eye(2))
        vals[i] = np.linalg.det(fish_mat)
    # reshape grid and vals for visualization
    xy = []
    for g in grid.T:
        xy.append(
            np.reshape(g, newshape=(50, 50))
        )
    vals = vals.reshape(50, 50)

    ######
    # second figure: visualize Fisher information
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.contourf(*xy, vals, levels=20)

    plt.show()
