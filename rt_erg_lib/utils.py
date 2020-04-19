import numpy as np
from math import pi
from numpy import sqrt
import matplotlib.pyplot as plt

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
    dm[1,0] = -1 / ( 1 + (alpha[1]-x[1])**2 / (alpha[0]-x[0])**2 ) * (alpha[1]-x[1]) / (alpha[0]-x[0])**2
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

def fisher_mat_broadcast(x, alpha, cov_inv):
    '''
    calculate FIM by broadcasting
    @x: N-by-2 array contain all grids
    '''
    # calculate derivative of measurement model
    dm1 = 0.5 / sqrt( (-x[:,0]+alpha[0])**2 + (-x[:,1]+alpha[1])**2 ) * 2 * (-x[:,0] + alpha[0])
    dm2 = 0.5 / sqrt( (-x[:,0]+alpha[0])**2 + (-x[:,1]+alpha[1])**2 ) * 2 * (-x[:,1] + alpha[1])
    dm3 = -1 / ( 1 + (-x[:,1]+alpha[1])**2 / (-x[:,0]+alpha[0])**2 ) * (-x[:,1]+alpha[1]) / (-x[:,0]+alpha[0])**2
    dm4 = 1 / ( 1 + (-x[:,1]+alpha[1])**2 / (-x[:,0]+alpha[0])**2 ) * (1) / (-x[:,0]+alpha[0])
    # calculate each element of FIM (4 in total)
    fim1 = dm1 * (dm1*cov_inv[0,0] + dm3*cov_inv[1,0]) + dm3 * (dm1*cov_inv[0,1] + dm3*cov_inv[1,1])
    fim2 = dm2 * (dm1*cov_inv[0,0] + dm3*cov_inv[1,0]) + dm4 * (dm1*cov_inv[0,1] + dm3*cov_inv[1,1])
    fim3 = dm1 * (dm2*cov_inv[0,0] + dm4*cov_inv[1,0]) + dm3 * (dm2*cov_inv[0,1] + dm4*cov_inv[1,1])
    fim4 = dm2 * (dm2*cov_inv[0,0] + dm4*cov_inv[1,0]) + dm4 * (dm2*cov_inv[0,1] + dm4*cov_inv[1,1])
    # calculate determinant of each FIM
    det = fim1 * fim3 - fim2 * fim4
    # filter out based on dist
    dist = (x[:,0]-alpha[0])**2 + (x[:,1]-alpha[1])**2
    dist_flag = dist < 16
    dist_flag = dist_flag.astype(int)
    det = det * dist_flag
    # return
    return det

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

def evaluation(logs, init_dist=None):
    '''
    automatic visualize/evaluate/compare results from different simulations
    @logs: 1-D array containing log data returned from simulation.start() method
    @init_dist:
    '''
    # initialization
    num_sims = len(logs)
    fig = plt.figure()
    tf = logs[0]['tf']
    taxis = np.arange(tf)
    # evaluation
    if init_dist is None:
        # first row: comparison of ergodic cost between true trajectory and estimated trajectory for each simulation
        axes_row_1 = []
        for i in range(num_sims):
            axes_row_1.append(fig.add_subplot(2, num_sims, i+1))
            axes_row_1[i].plot(taxis, logs[i]['metric_true'], c='b')
            axes_row_1[i].plot(taxis, logs[i]['metric_est'], c='r')
            axes_row_1[i].set_xlabel('Time Steps')
            axes_row_1[i].set_ylabel('Ergodic Metric')
            axes_row_1[i].set_title('Simulation ' + str(i+1))
            axes_row_1[i].legend(['True Trajectory', 'Estimated Trajectory'])

        # second row: uncertainty plot and estimation error plot
        axes_row_21 = fig.add_subplot(2, num_sims, num_sims+1)
        for i in range(num_sims):
            axes_row_21.plot(taxis, logs[i]['uncertainty'])
        axes_row_21.set_xlabel('Time Steps')
        axes_row_21.set_ylabel('Robot State Uncertainty')
        axes_row_21.set_title('Robot State Uncertainty Plot')
        axes_row_21.legend(['Simulation '+str(i+1) for i in range(num_sims)])

        axes_row_22 = fig.add_subplot(2, num_sims, num_sims+2)
        for i in range(num_sims):
            axes_row_22.plot(taxis, logs[i]['error'])
        axes_row_22.set_xlabel('Time Steps')
        axes_row_22.set_ylabel('State Estimation Error')
        axes_row_22.set_title('State Estimation Error Plot')
        axes_row_22.legend(['Simulation '+str(i+1) for i in range(num_sims)])

        axes_row_23 = fig.add_subplot(2, num_sims, num_sims+3)
        for i in range(num_sims):
            axes_row_23.plot(taxis, logs[i]['metric_error'])
        axes_row_23.set_xlabel('Time Steps')
        axes_row_23.set_ylabel('Ergodic Metric Error')
        axes_row_23.set_title('Ergodic Metric Error Plot')
        axes_row_23.legend(['Simulation '+str(i+1) for i in range(num_sims)])

    else:
        pass

    # finally, plot all of them :)
    plt.show()
