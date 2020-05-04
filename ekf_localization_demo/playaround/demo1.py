"""
Testing EKF-SLAM simulation
"""

import sys
sys.path.append("/home/msun/Code/ErgodicBSP")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.dynamic_target_dist2 import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.ekf_localization_sim2 import simulation_slam
import autograd.numpy as np
from math import sin, cos
from math import pi

###################################
# initialization
###################################

tf = 2000
size = 15.0
init_state = np.array([5., 5., 0.0])
sensor_range = 2
# motion_noise = np.array([0.02, 0.02, 0.01])
motion_noise = np.array([0.02, 0.02, 0.005])
measure_noise = np.array([0.005, 0.005])
mcov = np.diag(measure_noise)

# landmarks = np.load('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/uniform_2.npy')
# landmarks = np.load('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_4.npy')
landmarks = np.load('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_7.npy')

means = [np.array([3.5, 11.5]), np.array([11.5, 3.5])]
vars = [np.array([0.8, 0.8])**2, np.array([0.8, 0.8])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size, landmarks=landmarks, mcov=mcov, sensor_range=sensor_range)
# t_dist.update0()

###################################
# simulation 1
###################################

envTrue1 = IntegratorSE2(size=size)
modelTrue1 = IntegratorSE2(size=size)
envDR1 = IntegratorSE2(size=size)
modelDR1 = IntegratorSE2(size=size)

ergCtrlTrue1 = RTErgodicControl(modelTrue1, t_dist, horizon=100, num_basis=20, batch_size=200)
ergCtrlTrue1.phik = convert_phi2phik(ergCtrlTrue1.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlTrue1.init_phik = convert_phi2phik(ergCtrlTrue1.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR1 = RTErgodicControl(modelDR1, t_dist, horizon=100, num_basis=20, batch_size=200)
ergCtrlDR1.phik = convert_phi2phik(ergCtrlDR1.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR1.init_phik = convert_phi2phik(ergCtrlTrue1.basis, t_dist.grid_vals, t_dist.grid)

erg_ctrl_sim1 = simulation_slam(size, init_state, t_dist, modelTrue1, ergCtrlTrue1, envTrue1, modelDR1, ergCtrlDR1, envDR1, tf, landmarks, sensor_range, motion_noise, measure_noise)

log1 = erg_ctrl_sim1.start(report=True, debug=True, update=0, update_threshold=1e-5)
erg_ctrl_sim1.animate(point_size=1, show_traj=True, title='EKF Localization Test')

'''
###################################
# simulation 2
###################################

envTrue2 = IntegratorSE2(size=size)
modelTrue2 = IntegratorSE2(size=size)
envDR2 = IntegratorSE2(size=size)
modelDR2 = IntegratorSE2(size=size)

ergCtrlTrue2 = RTErgodicControl(modelTrue2, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlTrue2.phik = convert_phi2phik(ergCtrlTrue2.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlTrue2.init_phik = convert_phi2phik(ergCtrlTrue2.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR2 = RTErgodicControl(modelDR2, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlDR2.phik = convert_phi2phik(ergCtrlDR2.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR2.init_phik = convert_phi2phik(ergCtrlTrue2.basis, t_dist.grid_vals, t_dist.grid)

erg_ctrl_sim2 = simulation_slam(size, init_state, t_dist, modelTrue2, ergCtrlTrue2, envTrue2, modelDR2, ergCtrlDR2, envDR2, tf, landmarks, sensor_range, motion_noise, measure_noise)

log2 = erg_ctrl_sim2.start(report=True, debug=False, update=1, update_threshold=1e-3)

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

log3 = erg_ctrl_sim3.start(report=True, debug=False, update=2, update_threshold=1e-3)
# erg_ctrl_sim3.animate3(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)

###################################
# evaluation
###################################

evaluation([log1, log2, log3])
'''
