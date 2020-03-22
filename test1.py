import sys
from rt_erg_lib.double_integrator import DoubleIntegrator
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation import simulation
import autograd.numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = 10.0
env = DoubleIntegrator(size=size)
model = DoubleIntegrator(size=size)
means = [np.array([4.5, 2.5]), np.array([6.5, 8.5])]
vars = [np.array([0.7,0.7])**2, np.array([0.4,0.4])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
erg_ctrl = RTErgodicControl(model, t_dist, horizon=25, num_basis=15, batch_size=100)
erg_ctrl.phik = convert_phi2phik(basis=erg_ctrl.basis,
                                 phi_val=t_dist.grid_vals,
                                 phi_grid=t_dist.grid)

"""start simulation"""
tf = 2000
init_state = np.array([5, 5, 0, 0])
erg_ctrl_sim = simulation(size, init_state, t_dist, model, erg_ctrl, env, tf)
erg_ctrl_sim.start(report=True)
erg_ctrl_sim.animate(point_size=5, show_traj=True)
erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()