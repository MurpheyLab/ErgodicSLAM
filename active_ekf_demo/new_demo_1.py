"""
Testing EKF-SLAM simulation
"""

import sys
sys.path.append("/home/msun/Code/ErgodicBSP")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.active_target_dist2 import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.active_ekf_simulation2 import simulation_slam
import autograd.numpy as np
from math import sin, cos
from math import pi

"""initialization"""
tf = 200
size = 20.0
# size = 25.0
noise = 0.005
init_state = np.array([11., 2., 0.0])
envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

# generete target trajectory
means = [np.array([14.5, 5.5]), np.array([6.5, 15.5])]
# means = [np.array([10.5, 5.5]), np.array([10.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

erg_horizon = 20

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=erg_horizon, num_basis=15, batch_size=200)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=erg_horizon, num_basis=15, batch_size=200)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR.init_phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
# initialize user-defined landmarks
# landmarks = np.array([[14.3, 5.5],
#                       [13.9, 5.9]])
# landmarks = np.array([[9.5, 10.5],
#                       [10.5, 9.5]])

# lanmark distribution 1: uniform
landmarks1 = np.random.uniform(0.5, 19.5, size=(7, 2))
landmarks2 = np.random.uniform(0.5, 19.5, size=(8, 2))
landmarks = np.concatenate((landmarks1, landmarks2))

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

landmarks = np.random.uniform(0.5, 19.5, size=(30,2))

sensor_range = 5
motion_noise = np.array([0.04, 0.04, 0.01])
measure_noise = np.array([0.01, 0.01])
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, modelTrue, ergCtrlTrue, envTrue, modelDR, ergCtrlDR, envDR, tf, landmarks, sensor_range, motion_noise, measure_noise, switch=100)
erg_ctrl_sim.start(report=True, debug=False, update=3)

# erg_ctrl_sim.animate_eval(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)
# erg_ctrl_sim.animate2(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)
erg_ctrl_sim.animate(point_size=1, show_traj=True, plan=False, title='Landmarks Distribution')

# erg_ctrl_sim.plot(point_size=1, save=None)
# erg_ctrl_sim.path_reconstruct(save=None)
