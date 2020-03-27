"""
Testing EKF-SLAM simulation
"""

import sys
sys.path.append("/home/msun/Code/rt_ergodic_control")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.ekf_simulation import simulation_slam
import autograd.numpy as np

"""initialization"""
size = 20.0
# size = 25.0
noise = 0.005
init_state = np.array([11., 7., 0.0])
envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

means = [np.array([14.5, 5.5]), np.array([6.5, 15.5])]
# means = [np.array([10.5, 5.5]), np.array([10.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=100, num_basis=15, batch_size=200)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 2000
# lanmark distribution 1: uniform
# landmarks1 = np.random.uniform(0.5, 19.5, size=(10, 2))
# landmarks2 = np.random.uniform(0.5, 19.5, size=(10, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

# lanmark distribution 2: gathered at center
# landmarks1 = np.random.uniform(8.5, 15.0, size=(10, 2))
# landmarks2 = np.random.uniform(6.0, 13.5, size=(10, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

# lanmark distribution 3: gathered at two corners
landmarks1 = np.random.uniform(11.5, 19.0, size=(10, 2))
landmarks2 = np.random.uniform(1.0, 9.5, size=(10, 2))
landmarks = np.concatenate((landmarks1, landmarks2))

# landmark distribution 4
# landmarks = np.array([
#     [2.4, 10.1],
#     [17.4, 3.0],
#     [4.5, 6.5],
#     [12.5, 9.8],
#     [1.9, 17.9],
#     [9.1, 2.3],
#     [12.3, 14.5],
#     [17.8, 13.1],
#     [4.9, 9.0]
# ])

sensor_range = 4
motion_noise = np.array([0.2, 0.2, 0.1]) ** 2
# motion_noise = np.zeros(3)
measure_noise = np.array([0.1, 0.1]) ** 2
# measure_noise = np.zeros(2)
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, modelTrue, ergCtrlTrue, envTrue, modelDR, ergCtrlDR, envDR, tf, landmarks, sensor_range, motion_noise, measure_noise)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=1, show_traj=True)
erg_ctrl_sim.plot(point_size=1, save=None)
erg_ctrl_sim.path_reconstruct(save=None)