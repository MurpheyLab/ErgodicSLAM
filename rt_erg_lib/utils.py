import numpy as np
from math import pi, sqrt

def convert_phi2phik(basis, phi_val, phi_grid=None):
    '''
    Converts the distribution to the fourier decompositions
    '''
    if len(phi_val.shape) != 1:
        phi_val = phi_val.ravel()
    if phi_grid is None:
        print('--Assuming square grid')
        phi_grid = np.meshgrid(*[np.linspace(0, 1., int(np.sqrt(len(phi_val))))
                                for _ in range(2)])
        phi_grid = np.c_[phi_grid[0].ravel(), phi_grid[1].ravel()]
    assert phi_grid.shape[0] == phi_val.shape[0], 'samples are not the same'

    return np.sum([basis.fk(x) * v for v, x in zip(phi_val, phi_grid)], axis=0)

def convert_phik2phi(basis, phik, phi_grid=None):
    '''
    Reconstructs phi from the Fourier terms
    '''
    if phi_grid is None:
        print('--Assuming square grid')
        phi_grid = np.meshgrid(*[np.linspace(0, 1.)
                                for _ in range(2)])
        phi_grid = np.c_[phi_grid[0].ravel(), phi_grid[1].ravel()]
    phi_val = np.stack([np.dot(basis.fk(x), phik) for x in phi_grid])
    return phi_val

def convert_traj2ck(basis, xt):
    '''
    This utility function converts a trajectory into its time-averaged
    statistics in the Fourier domain
    '''
    N = len(xt)
    return np.sum([basis.fk(x) for x in xt], axis=0) / N

# 2020-03-01: add "size" parameter to support customizable exploration area size
def convert_ck2dist(basis, ck, grid=None, size=1.):
    '''
    This utility function converts a ck into its time-averaged
    statistics
    '''
    if grid is None:
        print('--Assuming square grid')
        # grid = np.meshgrid(*[np.linspace(0, 1.)
        #                         for _ in range(2)])
        grid = np.meshgrid(*[np.linspace(0, size)
                             for _ in range(2)])
        grid = np.c_[grid[0].ravel(), grid[1].ravel()]

    val = np.stack([np.dot(basis.fk(x), ck) for x in grid])
    return val

def normalize_angle(angle):
    '''
    normalize angle into [-pi, pi]
    '''
    normAngle = angle
    while normAngle > pi:
        normAngle -= 2*pi
    while normAngle < -pi:
        normAngle += 2*pi

    return normAngle

def multi_gaussian(x, mean, var):
    p = 1 / np.sqrt(np.linalg.det(2 * pi * var)) * np.exp(
                    -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(var)), (x - mean)))
    return p

def derivative_measure(x, alpha):
    '''
    @x: robot state (x1, x2, theta)
    @alpha: landmark state (alpha1, alpha2)
    '''
    dm = np.zeros((2,2))
    dm[0,0] = 0.5 / sqrt((alpha[0]-x[0])**2 + (alpha[1]-x[1])**2) * 2 * (alpha[0]-x[0])
    dm[0,1] = 0.5 / sqrt((alpha[0]-x[0])**2 + (alpha[1]-x[1])**2) * 2 * (alpha[1]-x[1])
    dm[1,0] = -1 / ( 1 + (alpha[1]-x[1])**2 / (alpha[0]-x[0])**2 ) * (alpha[1]-x[1]) /         (alpha[0]-x[0])**2
    dm[1,1] = 1 / ( 1 + (alpha[1]-x[1])**2 / (alpha[0]-x[0])**2 ) * (1) / (alpha[0]-x[0])
    return dm

def fisher_mat(x, alpha, cov_inv):
    '''
    @cov_inv: inverse of measurement noise covariance matrix
    '''
    dist = sqrt( (x[0]-alpha[0])**2 + (x[1]-alpha[1])**2 )
    if dist > 4:
        return np.zeros((2,2))
    else:
        dm = derivative_measure(x, alpha)
        return np.dot(np.dot(dm.T, cov_inv), dm)

def fisher_mat_expectation(x, alpha, cov_inv):
    '''
    @cov_inv: inverse of measurement noise covariance matrix
    '''
    dm = derivative_measure(x, alpha)
    return np.dot(np.dot(dm.T, cov_inv), dm)

def sample_expectation(rv, mean, cov, num=1000):
    val = 0
    sx, sy = np.random.multivariate_normal(mean, cov, num).T
    for i in range(sx.shape[0]):
        val += rv(sx[i], sy[i])
    val /= num
    return val
