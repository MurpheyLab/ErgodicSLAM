"""
Testing EKF-SLAM simulation
"""

import sys
sys.path.append("/home/msun/Code/ErgodicBSP")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.active_target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.active_ekf_simulation import simulation_slam
import autograd.numpy as np
from math import sin, cos
from math import pi

"""initialization"""
tf = 500
size = 20.0
# size = 25.0
noise = 0.005
# init_state = np.array([11., 7., 0.0])
# init_state = np.array([4., 5., 0.])
init_state = np.array([10., 1., 0])
envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

# generete target trajectory
means = [np.array([14.5, 5.5]), np.array([6.5, 15.5])]
# means = [np.array([10.5, 5.5]), np.array([10.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR.init_phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
# initialize user-defined landmarks
# landmarks = np.array([[14.3, 5.5],
#                       [13.9, 5.9]])
# landmarks = np.array([[9.5, 10.5],
#                       [10.5, 9.5]])

# lanmark distribution 1: uniform
# landmarks1 = np.random.uniform(0.5, 19.5, size=(7, 2))
# landmarks2 = np.random.uniform(0.5, 19.5, size=(8, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

# lanmark distribution 2: gathered at two corners
# landmarks1 = np.random.uniform(14.0, 18.0, size=(7, 2))
# landmarks2 = np.random.uniform(2.0, 6.0, size=(8, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

# lanmark distribution 3: mixed distribution
# landmarks1 = np.random.uniform(12.0, 18.0, size=(10, 2))
# landmarks2 = np.random.uniform(0.5, 19.5, size=(10, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

# read landmarks from file
# landmarks = np.load('landmarks_temp.npy')

# np.save("landmarks_temp.npy", landmarks)

# squarely arranged landmarks
landmarks = []
for x in np.linspace(1, size-1, int(size/2)):
    landmarks.append([x, 1])
    landmarks.append([x, size-1])
    landmarks.append([1, x])
    landmarks.append([size-1, x])
landmarks = np.array(landmarks)

sensor_range = 4
motion_noise = np.array([0.2, 0.15, 0.1]) ** 2
# motion_noise = np.array([0.35, 0.25, 0.15]) ** 2
# motion_noise = np.zeros(3)
measure_noise = np.array([0.15, 0.1]) ** 2
# measure_noise = np.array([1e-03, 1e-03]) ** 2
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, modelTrue, ergCtrlTrue, envTrue, modelDR, ergCtrlDR, envDR, tf, landmarks, sensor_range, motion_noise, measure_noise)
erg_ctrl_sim.start(report=True, debug=False, update=2)


erg_ctrl_sim.plot(point_size=1, save=None)

erg_ctrl_sim.path_reconstruct(save=None)

# erg_ctrl_sim.animate2(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)

erg_ctrl_sim.animate(point_size=2, alpha=4, show_traj=True, title='Landmarks Distribution Test', rate=50)

