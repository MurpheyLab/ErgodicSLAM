import sys
sys.path.append("..")
from rt_erg_lib.integrator_se2_hallway import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist_hallway import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation_ped_hallway import simulation
import autograd.numpy as np
import matplotlib.pyplot as plt

"""initialization"""
size = 10.0
env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)
means = [np.array([4.5, 0.8]), np.array([8.6,1.2])]
vars = [np.array([0.3, 0.3])**2, np.array([0.3,0.3])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
erg_ctrl = RTErgodicControl(model, t_dist, horizon=25, num_basis=20, batch_size=5)
# erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)

ped_data = np.load('sim_data_2.npy')
space = [
        np.array([(x, 0) for x in np.linspace(0, 10, 200)]),
        np.array([(x, 2) for x in np.linspace(0, 10, 200)]),
        np.array([( 0, y) for y in np.linspace(0, 2, 200)]),
        np.array([(10, y) for y in np.linspace(0, 2, 200)])
    ]

"""start simulation"""
tf = 1000
init_state = np.array([9., 0.2, 0.])
erg_ctrl_sim = simulation(size, init_state, model, erg_ctrl, env, tf, ped_data=ped_data, space=space, goals=None)
erg_ctrl_sim.start()
erg_ctrl_sim.animate2(point_size=80, show_traj=False)
# erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()
