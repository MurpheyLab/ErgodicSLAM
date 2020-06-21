"""
this is the 3rd version of testing ergodic control
the 1st version is kept as a good reference, this version
we can play with some parameters and new test data

this version updated distance field (including walls)
"""

import sys
sys.path.append('/home/msun/Code/ErgodicBSP')
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.simulation_ped_2 import simulation
import autograd.numpy as np
import matplotlib.pyplot as plt

robot_type = 1

"""initialization"""
size = 15.0
env = None
model = None
if robot_type == 1:
    from rt_erg_lib.integrator_se2 import IntegratorSE2
    env = IntegratorSE2(size=size)
    model = IntegratorSE2(size=size)
    init_state = np.array([*np.random.uniform(1, size-1, 2), 0])
    # init_state = np.array([13., 2., 0])
elif robot_type == 2:
    from rt_erg_lib.double_integrator import DoubleIntegrator
    env = DoubleIntegrator(size=size)
    model = DoubleIntegrator(size=size)
    # init_state = np.array([10.5, 4.5, 0, 0])
    init_state = np.array([*np.random.uniform(1, size-1, 2), 0, 0])
else:
    pass

means = [np.array([8.5, 6.5]), np.array([6.5, 8.5]), np.array([6.2,5.8])]
vars = [np.array([0.9,0.9])**2, np.array([0.7,0.7])**2, np.array([0.5,0.5])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)
# erg_ctrl = RTErgodicControl(model, t_dist, horizon=40, num_basis=20, batch_size=50) # last of last time
erg_ctrl = RTErgodicControl(model, t_dist, horizon=250, num_basis=25, batch_size=300) # good one
# erg_ctrl = RTErgodicControl(model, t_dist, horizon=80, num_basis=25, batch_size=20)

erg_ctrl.phik = convert_phi2phik(basis=erg_ctrl.basis,
                                 phi_val=t_dist.grid_vals,
                                 phi_grid=t_dist.grid)

ped_data = np.load('/home/msun/Code/ErgodicBSP/ped_sim/sim_data_5.npy')


space = [
        np.array([(x, 0) for x in np.linspace(0, 15, 200)]),
        np.array([(x, 15) for x in np.linspace(0, 15, 200)]),
        np.array([( 0, y) for y in np.linspace(0, 15, 200)]),
        np.array([(15, y) for y in np.linspace(0, 15, 200)])
    ]

"""start simulation"""
goals = np.array([[2., 13.], [13., 2.]])
tf = 500
erg_ctrl_sim = simulation(size, init_state, model, erg_ctrl, env, tf, ped_data=ped_data, space=space, goals=goals)
erg_ctrl_sim.start()
erg_ctrl_sim.animate2(point_size=20, show_traj=False, rate=20)
erg_ctrl_sim.plot(point_size=1)
erg_ctrl_sim.path_reconstruct()
