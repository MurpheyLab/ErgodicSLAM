"""
Testing EKF-SLAM simulation with same landmarks (but different
    observations)
"""

import sys
sys.path.append("..")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.ekf_mpc_simulation4 import simulation_slam
import autograd.numpy as np

"""initialization"""
size = 20.0
# size = 25.0
noise = 0.005
init_state = np.array([11., 2., 0.0])
envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

means = [np.array([14.5, 5.5]), np.array([6.5, 15.5])]
# means = [np.array([10.5, 5.5]), np.array([10.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

horizon = 150
batch_size = 250
num_basis = 15

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=horizon, num_basis=num_basis, batch_size=batch_size)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=horizon, num_basis=num_basis, batch_size=batch_size)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 1000
# lanmark distribution 1: uniform
# landmarks1 = np.random.uniform(0.5, 7.5, size=(6, 2))
# landmarks2 = np.random.uniform(12.5, 19.5, size=(4, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

landmarks = np.random.uniform(0.5, 19.5, size=(15, 2))
# landmarks = np.array([[6., 14.]])

sensor_range = 5
motion_noise = np.array([0.04, 0.04, 0.01])
measure_noise = np.array([0.01, 0.01])
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, modelTrue, ergCtrlTrue, envTrue, modelDR, ergCtrlDR, envDR, tf, landmarks, sensor_range, motion_noise, measure_noise, static_test=45, horizon=3, switch=1)
erg_ctrl_sim.start(report=True, debug=True)
erg_ctrl_sim.animate(point_size=1, show_traj=True, plan=False, title='Landmarks Distribution')
# erg_ctrl_sim.plot(point_size=1, save=None)
# erg_ctrl_sim.path_reconstruct(save=None)
# erg_ctrl_sim.static_test_plot(point_size=1, save=None)
