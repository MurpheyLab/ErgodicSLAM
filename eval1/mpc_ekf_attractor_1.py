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
from rt_erg_lib.ekf_mpc_simulation6 import simulation_slam
import autograd.numpy as np

"""initialization"""
size = 20.0
# size = 25.0
noise = 0.005
# init_state = np.array([11., 2., 0.0])
init_state = np.array([2., 2., 0.])
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
tf = 500
# lanmark distribution 1: cornered
landmarks1 = np.random.uniform(0.5, 3.5, size=(5, 2))
landmarks2 = np.random.uniform(16.5, 19.5, size=(5, 2))
landmarks = np.concatenate((landmarks1, landmarks2))

# landmarks = np.random.uniform(0.5, 19.5, size=(10, 2))

sensor_range = 5
motion_noise = np.array([0.04, 0.04, 0.01])
measure_noise = np.array([0.01, 0.01])
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, modelTrue, ergCtrlTrue, envTrue, modelDR, ergCtrlDR, envDR, tf, landmarks, sensor_range, motion_noise, measure_noise, static_test=45, horizon=3, switch=1, num_pts=50, stm_threshold=0.5)
log = erg_ctrl_sim.start(report=True, debug=True)
erg_ctrl_sim.animate2(point_size=1, show_traj=True, plan=False, title='Landmarks Distribution')
# erg_ctrl_sim.plot(point_size=1, save=None)
# erg_ctrl_sim.path_reconstruct(save=None)
# erg_ctrl_sim.static_test_plot(point_size=1, save=None)

fig = plt.figure()

ax1 = fig.add_subplot(231)
ax1.set_title('Pose Uncertainty')
ax1.plot(np.arange(tf)*0.1, log['pose_uncertainty'])

ax2 = fig.add_subplot(232)
ax2.set_title('Pose Est Error')
ax2.plot(np.arange(tf)*0.1, log['pose_err'])

ax3 = fig.add_subplot(233)
ax3.set_title('Landmark Avg Est Error')
ax3.plot(np.arange(tf)*0.1, log['lm_avg_err'])

ax4 = fig.add_subplot(234)
ax4.set_title('Actual Area Coverage')
ax4.plot(np.arange(tf)*0.1, log['true_area_coverage'])

ax5 = fig.add_subplot(235)
ax5.set_title('Est Area Coverage')
ax5.plot(np.arange(tf)*0.1, log['est_area_coverage'])

ax6 = fig.add_subplot(236)
ax6.set_title('Landmark Coverage')
ax6.plot(np.arange(tf)*0.1, log['landmark_coverage'])

plt.show()

