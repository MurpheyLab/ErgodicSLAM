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

tf = 2000
size = 20.0
init_state = np.array([10., 10., 0.0])
sensor_range = 4
motion_noise = np.array([0.2, 0.2, 0.1]) ** 2
measure_noise = np.array([0.15, 0.15]) ** 2
threshold = 1e-3

means = [np.array([14.5, 5.5]), np.array([4.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

landmarks = np.load('/home/msun/Code/ErgodicBSP/lm_dist_evaluation/cornered_dual_2.npy')

eval_time = 5
eval_logs = []
for idx in range(eval_time):
    print('evalulation round: ', idx)

    ###################################
    # simulation 1
    ###################################

    envTrue1 = IntegratorSE2(size=size)
    modelTrue1 = IntegratorSE2(size=size)
    envDR1 = IntegratorSE2(size=size)
    modelDR1 = IntegratorSE2(size=size)
    t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

    ergCtrlTrue1 = RTErgodicControl(modelTrue1, t_dist, horizon=100, num_basis=15, batch_size=200)
    ergCtrlTrue1.phik = convert_phi2phik(ergCtrlTrue1.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlTrue1.init_phik = convert_phi2phik(ergCtrlTrue1.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlDR1 = RTErgodicControl(modelDR1, t_dist, horizon=100, num_basis=15, batch_size=200)
    ergCtrlDR1.phik = convert_phi2phik(ergCtrlDR1.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlDR1.init_phik = convert_phi2phik(ergCtrlDR1.basis, t_dist.grid_vals, t_dist.grid)

    erg_ctrl_sim1 = simulation_slam(size, init_state, t_dist, modelTrue1, ergCtrlTrue1, envTrue1, modelDR1, ergCtrlDR1, envDR1, tf, landmarks, sensor_range, motion_noise, measure_noise)

    log1 = erg_ctrl_sim1.start(report=True, debug=False, update=0, update_threshold=threshold)
    # erg_ctrl_sim1.animate()
    # erg_ctrl_sim1.new_animate3(point_size=1, alpha=3, show_traj=True, title='Landmarks Distribution Test', rate=50)

    ###################################
    # simulation 2
    ###################################

    envTrue2 = IntegratorSE2(size=size)
    modelTrue2 = IntegratorSE2(size=size)
    envDR2 = IntegratorSE2(size=size)
    modelDR2 = IntegratorSE2(size=size)
    t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

    ergCtrlTrue2 = RTErgodicControl(modelTrue2, t_dist, horizon=100, num_basis=15, batch_size=200)
    ergCtrlTrue2.phik = convert_phi2phik(ergCtrlTrue2.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlTrue2.init_phik = convert_phi2phik(ergCtrlTrue2.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlDR2 = RTErgodicControl(modelDR2, t_dist, horizon=100, num_basis=15, batch_size=200)
    ergCtrlDR2.phik = convert_phi2phik(ergCtrlDR2.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlDR2.init_phik = convert_phi2phik(ergCtrlDR2.basis, t_dist.grid_vals, t_dist.grid)

    erg_ctrl_sim2 = simulation_slam(size, init_state, t_dist, modelTrue2, ergCtrlTrue2, envTrue2, modelDR2, ergCtrlDR2, envDR2, tf, landmarks, sensor_range, motion_noise, measure_noise)

    log2 = erg_ctrl_sim2.start(report=True, debug=False, update=1, update_threshold=threshold)

    ###################################
    # simulation 3
    ###################################

    envTrue3 = IntegratorSE2(size=size)
    modelTrue3 = IntegratorSE2(size=size)
    envDR3 = IntegratorSE2(size=size)
    modelDR3 = IntegratorSE2(size=size)
    t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

    ergCtrlTrue3 = RTErgodicControl(modelTrue3, t_dist, horizon=100, num_basis=15, batch_size=200)
    ergCtrlTrue3.phik = convert_phi2phik(ergCtrlTrue3.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlTrue3.init_phik = convert_phi2phik(ergCtrlTrue3.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlDR3 = RTErgodicControl(modelDR3, t_dist, horizon=100, num_basis=15, batch_size=200)
    ergCtrlDR3.phik = convert_phi2phik(ergCtrlDR3.basis, t_dist.grid_vals, t_dist.grid)
    ergCtrlDR3.init_phik = convert_phi2phik(ergCtrlDR3.basis, t_dist.grid_vals, t_dist.grid)

    erg_ctrl_sim3 = simulation_slam(size, init_state, t_dist, modelTrue3, ergCtrlTrue3, envTrue3, modelDR3, ergCtrlDR3, envDR3, tf, landmarks, sensor_range, motion_noise, measure_noise)

    log3 = erg_ctrl_sim3.start(report=True, debug=False, update=2, update_threshold=threshold)
    # erg_ctrl_sim3.animate3(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)

    ###################################
    # evaluation
    ###################################

    eval_log = evaluation([log1, log2, log3], eval_id=idx, plot=False)
    eval_logs.append(eval_log)


###################################
# process averaged data
###################################
avg_metric = 0
avg_unc = 0
avg_metric_err = 0
avg_est_err = 0
for log in eval_logs:
    avg_metric += log['avg_metric']
    avg_unc += log['avg_unc']
    avg_metric_err += log['avg_metric_err']
    avg_est_err += log['avg_est_err']
avg_metric /= eval_time
avg_unc /= eval_time
avg_metric_err /= eval_time
avg_est_err /= eval_time


with open('final_report', 'w') as fr:

    print('\n\n****************************************')
    print('Final Report')
    print('****************************************\n')
    fr.write('\n\n****************************************\n')
    fr.write('Final Report\n')
    fr.write('****************************************\n\n')


    num_sims = avg_metric.shape[0]
    print('----------------------------------------')
    print('Time Averaged State Uncertainty')
    print('----------------------------------------')
    fr.write('----------------------------------------\n')
    fr.write('Time Averaged State Uncertainty\n')
    fr.write('----------------------------------------\n')
    for i in range(num_sims):
        print('\tSimulation {}: {}'.format(i+1, avg_unc[i]))
        fr.write('\tSimulation {}: {}\n'.format(i+1, avg_unc[i]))

    print('----------------------------------------')
    print('Time Averaged Estimation Error')
    print('----------------------------------------')
    fr.write('----------------------------------------\n')
    fr.write('Time Averaged Estimation Error\n')
    fr.write('----------------------------------------\n')
    for i in range(num_sims):
        print('\tSimulation {}: {}'.format(i+1, avg_est_err[i]))
        fr.write('\tSimulation {}: {}\n'.format(i+1, avg_est_err[i]))

    print('----------------------------------------')
    print('Time Averated Ergodic Metric')
    print('----------------------------------------')
    fr.write('----------------------------------------\n')
    fr.write('Time Averated Ergodic Metric\n')
    fr.write('----------------------------------------\n')
    for i in range(num_sims):
        print('\tSimulation {}: {}'.format(i+1, avg_metric[i]))
        fr.write('\tSimulation {}: {}\n'.format(i+1, avg_metric[i]))

    print('----------------------------------------')
    print('Time Averaged Ergodic Metric Error')
    print('----------------------------------------')
    fr.write('----------------------------------------\n')
    fr.write('Time Averaged Ergodic Metric Error\n')
    fr.write('----------------------------------------\n')
    for i in range(num_sims):
        print('\tSimulation {}: {}'.format(i+1, avg_metric_err[i]))
        fr.write('\tSimulation {}: {}\n'.format(i+1, avg_metric_err[i]))
    print('----------------------------------------')
    fr.write('----------------------------------------\n')
