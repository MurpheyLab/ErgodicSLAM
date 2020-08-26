import sys
sys.path.append("..")

from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation import simulation

import numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = 3.0
init_state = np.array([size/2, size/2, 0])
env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)

num_pts = 50
means = [np.array([0.5, 1.8]), np.array([2.6,0.6]), np.array([2.5, 2.5])]
vars = [np.array([0.2,0.2])**2, np.array([0.2,0.2])**2, np.array([0.2,0.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

erg_ctrl = RTErgodicControl(model, t_dist, horizon=50, num_basis=15, batch_size=-1)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 2000
erg_ctrl_sim = simulation(size, init_state, t_dist, model, erg_ctrl, env, tf)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=5, show_traj=True)
erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()

