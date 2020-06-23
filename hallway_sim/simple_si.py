import sys
sys.path.append("/home/msun/Code/ErgodicBSP")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation import simulation
import autograd.numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = 10.0
env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)
means = [np.array([4.5, 0.8]), np.array([8.6,1.2])]
vars = [np.array([0.3, 0.3])**2, np.array([0.3,0.3])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
erg_ctrl = RTErgodicControl(model, t_dist, horizon=50, num_basis=15, batch_size=100)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
tf = 1000
init_state = np.array([8., 0.2, 0.])
erg_ctrl_sim = simulation(size, init_state, t_dist, model, erg_ctrl, env, tf)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=5, show_traj=False)
erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()
