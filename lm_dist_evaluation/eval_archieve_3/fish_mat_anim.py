"""
Testing EKF-SLAM simulation
"""

import sys
sys.path.append("/home/msun/Code/ErgodicBSP")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.dynamic_target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.dynamic_target_simulation import simulation_slam
import autograd.numpy as np
from math import sin, cos
from math import pi

###################################
# initialization
###################################

tf = 1000
size = 10.0
# init_state = np.array([11., 7., 0.0])
init_state = np.array([5., 5., 0.0])
sensor_range = 2.5
motion_noise = np.array([0.1, 0.1, 0.1]) ** 2
measure_noise = np.array([0.1, 0.1]) ** 2

means = [np.array([3.,7.]), np.array([7., 3.])]
vars = [np.array([.5, .5])**2, np.array([.5, .5])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

landmarks = np.load('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/two_diag.npy')

###################################
# simulation 3
###################################

envTrue3 = IntegratorSE2(size=size)
modelTrue3 = IntegratorSE2(size=size)
envDR3 = IntegratorSE2(size=size)
modelDR3 = IntegratorSE2(size=size)

ergCtrlTrue3 = RTErgodicControl(modelTrue3, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlTrue3.phik = convert_phi2phik(ergCtrlTrue3.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlTrue3.init_phik = convert_phi2phik(ergCtrlTrue3.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR3 = RTErgodicControl(modelDR3, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlDR3.phik = convert_phi2phik(ergCtrlDR3.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR3.init_phik = convert_phi2phik(ergCtrlTrue3.basis, t_dist.grid_vals, t_dist.grid)

erg_ctrl_sim3 = simulation_slam(size, init_state, t_dist, modelTrue3, ergCtrlTrue3, envTrue3, modelDR3, ergCtrlDR3, envDR3, tf, landmarks, sensor_range, motion_noise, measure_noise)

log3 = erg_ctrl_sim3.start(report=True, debug=True, update=0, update_threshold=1e-3)
erg_ctrl_sim3.plot(point_size=1, save=None)
erg_ctrl_sim3.animate(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)

