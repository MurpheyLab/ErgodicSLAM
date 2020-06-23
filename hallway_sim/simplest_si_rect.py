import sys
sys.path.append("/home/msun/Code/ErgodicBSP")
from rt_erg_lib.integrator_se2_rect import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist_rect import TargetDist
from rt_erg_lib.utils_rect import *
from rt_erg_lib.simulation_rect import simulation
import autograd.numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = np.array([10., 2.])
num_pts = 100
env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)
means = [np.array([3.0, 1.0]), np.array([7.0, 1.0])]
vars = [np.array([0.5, 0.5])**2 for _ in range(2)]
t_dist = TargetDist(num_pts=num_pts, means=means, vars=vars, size=size)


erg_ctrl = RTErgodicControl(model, t_dist, horizon=100, num_basis=15, batch_size=200)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid, size=size, num_pts=num_pts)

"""start simulation"""
tf = 500
init_state = np.array([5., 1., 0.])
erg_ctrl_sim = simulation(size, init_state, t_dist, model, erg_ctrl, env, tf)
erg_ctrl_sim.start(report=True)
# erg_ctrl_sim.animate(point_size=5, show_traj=False)
erg_ctrl_sim.plot(point_size=1)
# erg_ctrl_sim.path_reconstruct()
