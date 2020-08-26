import sys
sys.path.append('..')

from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist as StaticTargetDist
from rt_erg_lib.active_target_dist3 import TargetDist as DynTargetDist
from rt_erg_lib.utils import *

from rt_erg_lib.ekf_mpc_simulation6 import simulation_slam as ekf_mpc_simulation
from rt_erg_lib.active_ekf_simulation5 import simulation_slam as ekf_erg_simulation

import autograd.numpy as np
import matplotlib.pyplot as plt


"""initialization"""
tf = 50
size = 20.0
init_state = np.array([3.0, 3.0, 0.0])

landmarks1 = np.random.uniform(0.5, 3.5, size=(5, 2))
landmarks2 = np.random.uniform(16.5, 19.5, size=(5, 2))
landmarks = np.concatenate((landmarks1, landmarks2))

# landmarks = np.random.uniform(0.5, 19.5, size=(10,2))

sensor_range = 5
motion_noise = np.array([0.04, 0.04, 0.01])
measure_noise = np.array([0.01, 0.01])

"""ergodic ekf config"""

envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

means = [np.array([14.5, 5.5]), np.array([6.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = DynTargetDist(num_pts=50, means=means, vars=vars, size=size)

erg_horizon = 100
batch_size = -1

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=erg_horizon, num_basis=15, batch_size=batch_size)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=erg_horizon, num_basis=15, batch_size=batch_size)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR.init_phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

erg_sim = ekf_erg_simulation(size, init_state, t_dist, \
                             modelTrue, ergCtrlTrue, envTrue, \
                             modelDR, ergCtrlDR, envDR, \
                             tf, landmarks, sensor_range, \
                             motion_noise, measure_noise, switch=1)
erg_log = erg_sim.start(report=True, debug=False, update=7)

"""mpc ekf config"""

envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

means = [np.array([14.5, 5.5]), np.array([6.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = StaticTargetDist(num_pts=50, means=means, vars=vars, size=size)

erg_horizon = 100
batch_size = -1

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=erg_horizon, num_basis=15, batch_size=batch_size)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=erg_horizon, num_basis=15, batch_size=batch_size)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR.init_phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

mpc_sim = ekf_mpc_simulation(size, init_state, t_dist, \
                             modelTrue, ergCtrlTrue, envTrue, \
                             modelDR, ergCtrlDR, envDR, \
                             tf, landmarks, sensor_range, \
                             motion_noise, measure_noise, static_test=45, \
                             horizon=3, switch=1, num_pts=50, stm_threshold=0.5)
mpc_log = mpc_sim.start(report=True, debug=False)

"""print log""" # what's elapsed time here?
avg = lambda l : sum(l)/len(l)

print("====== ergodic control ======")
print("avg pose uncertainty: ", avg(erg_log['pose_uncertainty']))
print("avg pose error: ", avg(erg_log['pose_err']))
print("avg landmark error: ", avg(erg_log['lm_avg_err']))

print("====== model predictive control ======")
print("avg pose uncertainty: ", avg(mpc_log['pose_uncertainty']))
print("avg pose error: ", avg(mpc_log['pose_err']))
print("avg landmark error: ", avg(mpc_log['lm_avg_err']))


"""visualization"""

fig = plt.figure(1)

ax1 = fig.add_subplot(231)
ax1.set_title('Pose Uncertainty')
ax1.plot(np.arange(tf)*0.1, erg_log['pose_uncertainty'])
ax1.plot(np.arange(tf)*0.1, mpc_log['pose_uncertainty'])
ax1.legend(['erg', 'mpc'])

ax2 = fig.add_subplot(232)
ax2.set_title('Pose Est Error')
ax2.plot(np.arange(tf)*0.1, erg_log['pose_err'])
ax2.plot(np.arange(tf)*0.1, mpc_log['pose_err'])
ax2.legend(['erg', 'mpc'])

ax3 = fig.add_subplot(233)
ax3.set_title('Landmark Avg Est Error')
ax3.plot(np.arange(tf)*0.1, erg_log['lm_avg_err'])
ax3.plot(np.arange(tf)*0.1, mpc_log['lm_avg_err'])
ax3.legend(['erg', 'mpc'])

ax4 = fig.add_subplot(234)
ax4.set_title('Actual Area Coverage')
ax4.plot(np.arange(tf)*0.1, erg_log['true_area_coverage'])
ax4.plot(np.arange(tf)*0.1, mpc_log['true_area_coverage'])
ax4.legend(['erg', 'mpc'])

ax5 = fig.add_subplot(235)
ax5.set_title('Est Area Coverage')
ax5.plot(np.arange(tf)*0.1, erg_log['est_area_coverage'])
ax5.plot(np.arange(tf)*0.1, mpc_log['est_area_coverage'])
ax5.legend(['erg', 'mpc'])

ax6 = fig.add_subplot(236)
ax6.set_title('Landmark Coverage')
ax6.plot(np.arange(tf)*0.1, erg_log['landmark_coverage'])
ax6.plot(np.arange(tf)*0.1, mpc_log['landmark_coverage'])
ax6.legend(['erg', 'mpc'])

fig2 = plt.figure(2)
point_size = 10

ax1_fig2 = fig2.add_subplot(121)
ax1_fig2.set_title('Ergodic Trajectory')
ax1_fig2.set_aspect('equal')
ax1_fig2.set_xlim(0, size)
ax1_fig2.set_ylim(0, size)
ax1_fig2.scatter(landmarks[:,0], landmarks[:,1], c='black', marker='+')
xt_true = np.stack(erg_log['trajectory_true'])
xt_est = np.stack(erg_log['trajectory_slam'])
ax1_fig2.scatter(xt_true[:,0], xt_true[:,1], c='red', s=point_size)
ax1_fig2.scatter(xt_est[:,0], xt_est[:,1], c='blue', s=point_size)
ax1_fig2.legend(['Landmarks', 'True Traj', 'Est Traj'])

ax2_fig2 = fig2.add_subplot(122)
ax2_fig2.set_title('MPC Trajectory')
ax2_fig2.set_aspect('equal')
ax2_fig2.set_xlim(0, size)
ax2_fig2.set_ylim(0, size)
ax2_fig2.scatter(landmarks[:,0], landmarks[:,1], c='black', marker='+')
xt_true = np.stack(mpc_log['trajectory_true'])
xt_est = np.stack(mpc_log['trajectory_slam'])
ax2_fig2.scatter(xt_true[:,0], xt_true[:,1], c='red', s=point_size)
ax2_fig2.scatter(xt_est[:,0], xt_est[:,1], c='blue', s=point_size)
ax2_fig2.legend(['Landmarks', 'True Traj', 'Est Traj'])

plt.show()
