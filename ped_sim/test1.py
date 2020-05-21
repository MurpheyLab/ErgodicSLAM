import sys
sys.path.append('/home/msun/Code/ErgodicBSP')
# from rt_erg_lib.double_integrator import DoubleIntegrator
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation_ped import simulation
import autograd.numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = 15.0
env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)
means = [np.array([8.5, 6.5]), np.array([6.5, 8.5]), np.array([6.2,5.8])]
vars = [np.array([0.9,0.9])**2, np.array([0.7,0.7])**2, np.array([0.5,0.5])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
erg_ctrl = RTErgodicControl(model, t_dist, horizon=25, num_basis=20, batch_size=50)
# erg_ctrl = RTErgodicControl(model, t_dist, horizon=50, num_basis=25, batch_size=100)

erg_ctrl.phik = convert_phi2phik(basis=erg_ctrl.basis,
                                 phi_val=t_dist.grid_vals,
                                 phi_grid=t_dist.grid)

ped_data = np.load('/home/msun/Code/ErgodicBSP/ped_sim/sim_data_40.npy')

"""start simulation"""
tf = 200
init_state = np.array([6.5, 6.5, 0])
erg_ctrl_sim = simulation(size, init_state, t_dist, model, erg_ctrl, env, tf, ped_data=ped_data)
erg_ctrl_sim.start()
erg_ctrl_sim.animate(point_size=10, show_traj=False, rate=20)
erg_ctrl_sim.plot(point_size=1)
# erg_ctrl_sim.path_reconstruct()
