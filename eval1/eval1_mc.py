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
tf = 500
size = 20.0

sensor_range = 5
motion_noise = np.array([0.04, 0.04, 0.01])
measure_noise = np.array([0.01, 0.01])

num_mc = 4
erg_log = []
mpc_attractor_log = []
mpc_log = []

for exp_idx in range(num_mc):
    print('iter: {}'.format(exp_idx))

    """landmark generation"""
    landmarks1 = np.random.uniform(0.5, 3.5, size=(5, 2))
    landmarks2 = np.random.uniform(16.5, 19.5, size=(5, 2))
    landmarks = np.concatenate((landmarks1, landmarks2))

    # landmarks = np.random.uniform(0.5, 19.5, size=(10,2))

    init_state = np.array([3.0, 3.0, 0.0])
    # init_state = np.array([size/2., size/2., 0.])

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
    erg_log.append( erg_sim.start(report=True, debug=False, update=7) )

    """mpc ekf + atttractor config"""

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

    mpc_attractor_sim = ekf_mpc_simulation(size, init_state, t_dist, \
                                 modelTrue, ergCtrlTrue, envTrue, \
                                 modelDR, ergCtrlDR, envDR, \
                                 tf, landmarks, sensor_range, \
                                 motion_noise, measure_noise, static_test=45, \
                                 horizon=3, switch=1, num_pts=50, stm_threshold=0.5)
    mpc_attractor_log.append( mpc_attractor_sim.start(report=True, debug=False, no_attractor=False) )

    """mpc ekf no atttractor config"""

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
    mpc_log.append( mpc_sim.start(report=True, debug=False, no_attractor=True) )

"""compute average metrics"""
pose_uncertainty_mpc = 0
pose_err_mpc = 0
landmark_coverage_mpc = 0
ctrl_effort_mpc = 0
for log in mpc_log:
    pose_uncertainty_avg = np.sum(log['pose_uncertainty']) / len(log['pose_uncertainty'])
    pose_uncertainty_mpc += pose_uncertainty_avg
    pose_err_avg = np.sum(log['pose_err']) / len(log['pose_err'])
    pose_err_mpc += pose_err_avg
    landmark_coverage_mpc += log['landmark_coverage'][-1]
    ctrl_effort_avg = np.sum(log['ctrl_effort']) / len(log['ctrl_effort'])
    ctrl_effort_mpc += ctrl_effort_avg
pose_uncertainty_mpc /= num_mc * 1.0
pose_err_mpc /= num_mc * 1.0
landmark_coverage_mpc /= num_mc * 1.0
ctrl_effort_mpc /= num_mc * 1.0
print('\n------ mpc no attractor ------')
print('pose uncertainty: ', pose_uncertainty_mpc)
print('pose estimation err: ', pose_err_mpc)
print('landmark coverage: ', landmark_coverage_mpc)
print('control effort: ', ctrl_effort_mpc)


pose_uncertainty_mpca = 0
pose_err_mpca = 0
landmark_coverage_mpca = 0
ctrl_effort_mpca = 0
for log in mpc_attractor_log:
    pose_uncertainty_avg = np.sum(log['pose_uncertainty']) / len(log['pose_uncertainty'])
    pose_uncertainty_mpca += pose_uncertainty_avg
    pose_err_avg = np.sum(log['pose_err']) / len(log['pose_err'])
    pose_err_mpca += pose_err_avg
    landmark_coverage_mpca += log['landmark_coverage'][-1]
    ctrl_effort_avg = np.sum(log['ctrl_effort']) / len(log['ctrl_effort'])
    ctrl_effort_mpca += ctrl_effort_avg
pose_uncertainty_mpca /= num_mc * 1.0
pose_err_mpca /= num_mc * 1.0
landmark_coverage_mpca /= num_mc * 1.0
ctrl_effort_mpca /= num_mc * 1.0
print('\n------ mpc + attractor ------')
print('pose uncertainty: ', pose_uncertainty_mpca)
print('pose estimation err: ', pose_err_mpca)
print('landmark coverage: ', landmark_coverage_mpca)
print('control effort: ', ctrl_effort_mpca)


pose_uncertainty_erg = []
pose_err_erg = []
landmark_coverage_erg = []
ctrl_effort_erg = []
for log in erg_log:
    pose_uncertainty_avg = np.sum(log['pose_uncertainty']) / len(log['pose_uncertainty'])
    pose_uncertainty_erg.append( pose_uncertainty_avg )
    pose_err_avg = np.sum(log['pose_err']) / len(log['pose_err'])
    pose_err_erg.append( pose_err_avg )
    landmark_coverage_erg.append( log['landmark_coverage'][-1] )
    ctrl_effort_avg = np.sum(log['ctrl_effort']) / len(log['ctrl_effort'])
    ctrl_effort_erg.append( ctrl_effort_avg )

pose_uncertainty_erg = np.array(pose_uncertainty_erg)
pose_err_erg = np.array(pose_err_erg)
landmark_coverage_erg = np.array(landmark_coverage_erg)
ctrl_effort_erg = np.array(ctrl_effort_erg)

pose_uncertainty_erg_mean = pose_uncertainty_erg.mean()
pose_uncertainty_erg_std = pose_uncertainty_erg.std()
pose_err_erg_mean = pose_err_erg.mean()
pose_err_erg_std = pose_err_erg.std()
landmark_coverage_erg_mean = landmark_coverage_erg.mean()
landmark_coverage_erg_std = landmark_coverage_erg.std()
ctrl_effort_erg_mean = ctrl_effort_erg.mean()
ctrl_effort_erg_std = ctrl_effort_erg.std()

print('\n------ ergodic exploration ------')
print('pose uncertainty: {} +- {}'.format(pose_uncertainty_erg_mean, pose_uncertainty_erg_std))
print('pose estimation err: {} +- {}'.format(pose_err_erg_mean, pose_err_erg_std))
print('landmark coverage: {} +- {}'.format(landmark_coverage_erg_mean, landmark_coverage_erg_std))
print('control effort: {} +- {}'.format(ctrl_effort_erg_mean, ctrl_effort_erg_std))

