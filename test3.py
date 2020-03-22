"""
Testing noisy SLAM simulation
"""

import sys
sys.path.append("/home/msun/Code/rt_ergodic_control")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation_slam_noisy import simulation_slam
import autograd.numpy as np

"""initialization"""
size = 10.0
# size = 25.0
noise = 0.005
init_state = np.array([5., 5., 0.0])
envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

means = [np.array([4.5, 2.5]), np.array([6.5, 8.5])]
vars = [np.array([0.7, 0.7])**2, np.array([0.7, 0.7])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 2500
landmarks1 = np.random.uniform(0.2, 3.8, size=(6,2))
landmarks2 = np.random.uniform(8.2, 9.8, size=(4,2))
landmarks = np.concatenate((landmarks1, landmarks2))
sensor_range = 2
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, modelTrue, ergCtrlTrue, envTrue, modelDR, ergCtrlDR, envDR, tf, landmarks, sensor_range)
erg_ctrl_sim.start(noise=0.1, report=True)
erg_ctrl_sim.animate(point_size=1, show_label=True, show_traj=True)
erg_ctrl_sim.plot(point_size=1, save=None)
erg_ctrl_sim.path_reconstruct(save=None)