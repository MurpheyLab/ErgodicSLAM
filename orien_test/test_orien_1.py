import sys
sys.path.append("..")

from rt_erg_lib.integrator_se2_orien import IntegratorSE2
from rt_erg_lib.ergodic_control_orien import RTErgodicControl
from rt_erg_lib.target_dist_orien import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation_orien import simulation

# import autograd.numpy as np
import numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = 3.0
env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)

'''
means = [np.array([6.6, 4.7]), np.array([16.5,15.5]), np.array([4.4, 12.4])]
vars = [np.array([1.0,1.0])**2 for _ in range(3)]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
'''
means = [np.array([0.5, 1.8, 0.0]), np.array([2.6, 0.6, 0.0]), np.array([2.5, 2.5, 0.0])]
vars = [np.array([0.2, 0.2, 0.1])**2 for _ in range(3)]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

erg_ctrl = RTErgodicControl(model, t_dist, horizon=30, num_basis=10, batch_size=200)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 1000
init_state = np.array([size/2., size/2., 0.])
erg_ctrl_sim = simulation(size, init_state, t_dist, model, erg_ctrl, env, tf)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=5, show_traj=True, show_label=True)
erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()
