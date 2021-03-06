"""
Testing active EKF-SLAM simulation
use mutual-fisher information as target
try to make area coverage more reliable (problem in mi_11)
consider agent uncertainty in OG
add frontier attraction
landmarks: two cornered
params:
size = 20
erg_horizon = 100 (this has to be long for revisiting to happen)
batch_size = -1 (remain unknown what'a proper one)
tf = 2000 (usually reach full exploration around 1500)
fim_weight = 0.1
mi_weight = 0.1 (this can decrase ergodicity weight)
"""

import sys
sys.path.append("..")
from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.active_target_dist3 import TargetDist
from rt_erg_lib.utils import *
from rt_erg_lib.active_ekf_simulation4 import simulation_slam
import autograd.numpy as np
from math import sin, cos
from math import pi

"""initialization"""
tf = 2000
size = 20.0
# size = 25.0
noise = 0.005
# init_state = np.array([11., 2., 0.0]) # for reproduce mi_1: uniform lm
init_state = np.array([3., 3., 0.])
envTrue = IntegratorSE2(size=size)
modelTrue = IntegratorSE2(size=size)
envDR = IntegratorSE2(size=size)
modelDR = IntegratorSE2(size=size)

# generete target trajectory
means = [np.array([14.5, 5.5]), np.array([6.5, 15.5])]
# means = [np.array([10.5, 5.5]), np.array([10.5, 15.5])]
vars = [np.array([1.2, 1.2])**2, np.array([1.2, 1.2])**2]
t_dist = TargetDist(num_pts=50, means=means, vars=vars, size=size)

erg_horizon = 100
batch_size = -1

ergCtrlTrue = RTErgodicControl(modelTrue, t_dist, horizon=erg_horizon, num_basis=15, batch_size=batch_size)
ergCtrlTrue.phik = convert_phi2phik(ergCtrlTrue.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR = RTErgodicControl(modelDR, t_dist, horizon=erg_horizon, num_basis=15, batch_size=batch_size)
ergCtrlDR.phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)
ergCtrlDR.init_phik = convert_phi2phik(ergCtrlDR.basis, t_dist.grid_vals, t_dist.grid)

"""start simulation"""
# initialize user-defined landmarks
# landmarks = np.array([[14.3, 5.5],
#                       [13.9, 5.9]])
# landmarks = np.array([[9.5, 10.5],
#                       [10.5, 9.5]])

# lanmark distribution 1: uniform
# landmarks1 = np.random.uniform(0.5, 19.5, size=(7, 2))
# landmarks2 = np.random.uniform(0.5, 19.5, size=(8, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

# lanmark distribution 2: gathered at two corners
# landmarks1 = np.random.uniform(14.0, 18.0, size=(7, 2))
# landmarks2 = np.random.uniform(2.0, 6.0, size=(8, 2))
# landmarks = np.concatenate((landmarks1, landmarks2))

# lanmark distribution 3: mixed distribution
# landmarks1 = np.random.uniform(2.0, 6.0, size=(5, 2))
# landmarks2 = np.random.uniform(0.5, 0.5, size=(2, 2))
# landmarks2 = np.array([[0.5,19.5], [1.5,19.0], [19.5,0.5], [19.5,19.5], [10.,10.], [9.,5.], [4.,8.]])
# landmarks = np.concatenate((landmarks1, landmarks2))

# landmark distribution 4:square
# squarely arranged landmarks
'''
landmarks = []
for x in np.linspace(1, size-1, int(size/2)):
    landmarks.append([x, 1])
    landmarks.append([x, size-1])
    landmarks.append([1, x])
    landmarks.append([size-1, x])
landmarks = np.array(landmarks)
'''

# read landmarks from file
landmarks = np.load('landmarks_temp_cornered.npy')

# landmarks = np.random.uniform(0.5, size-0.5, size=(10,2))


# np.save("landmarks_temp_cornered.npy", landmarks)

sensor_range = 4
motion_noise = np.array([0.04, 0.04, 0.01])
measure_noise = np.array([0.01, 0.01])
erg_ctrl_sim = simulation_slam(size, init_state, t_dist, modelTrue, ergCtrlTrue, envTrue, modelDR, ergCtrlDR, envDR, tf, landmarks, sensor_range, motion_noise, measure_noise, switch=1)
erg_ctrl_sim.start(report=True, debug=False, update=8)

# erg_ctrl_sim.animate_eval(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)
# erg_ctrl_sim.animate2(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=50)
erg_ctrl_sim.animate3(point_size=1, alpha=1, show_traj=True, title='Landmarks Distribution Test', rate=100)
# erg_ctrl_sim.animate(point_size=1, show_traj=True, plan=False, title='Landmarks Distribution')

# erg_ctrl_sim.plot(point_size=1, save=None)
# erg_ctrl_sim.path_reconstruct(save=None)
