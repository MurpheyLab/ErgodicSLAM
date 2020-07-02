import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from tqdm import tqdm


size = 20.
num_lm = 20
landmarks = np.random.uniform(0.5, size-0.5, size=(num_lm, 2))

sensor_range = 4.
sensor_noise = np.array([0.02, 0.02]) # std dev of noise
motion_noise = np.array([0.02, 0.02, 0.01])

dt = 0.1
tf = 1000

init_state = np.array([size/2, 2., 0.])

est_traj = [init_state]
true_traj = [init_state]

def controller(x):
    return np.array([0.5, 0.2])

def g(x, u): # process model (x_dot)
    xdot = np.zeros(x.shape[0])
    xdot[0] = cos(x[2]) * u[0]
    xdot[1] = sin(x[2]) * u[0]
    xdot[2] = u[1]
    return xdot

def dgdx(x, u):
    N = x.shape[0]
    dg = np.zeros((N, N))
    dg[0][2] = -sin(x[2]) * u[0]
    dg[1][2] =  cos(x[2]) * u[0]
    return dg

def normalize(b):
    return (b + pi) % (2*pi) - pi

def obs(x, landmarks):
    r = x[0:3].copy()
    lm_x = landmarks[:,0].copy()
    lm_y = landmarks[:,1].copy()

    diff_x = lm_x - r[0]
    diff_y = lm_y - r[1]

    obs_rg = np.sqrt(diff_x**2 + diff_y**2)
    obs_br = np.arctan2(diff_y, diff_x) - r[2]

    obs_rg *= (obs_rg <= sensor_range).astype(float)
    obs_br *= (obs_rg <= sensor_range).astype(float)

    obs = np.stack((obs_rg, obs_br))
    return obs

state = init_state
for t in tqdm(range(tf)):
    ctrl = controller(state)

