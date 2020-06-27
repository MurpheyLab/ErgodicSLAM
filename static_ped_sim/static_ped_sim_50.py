import sys
sys.path.append('..')

from rt_erg_lib.integrator_se2 import IntegratorSE2
from rt_erg_lib.ergodic_control import RTErgodicControl
from rt_erg_lib.target_dist_static_ped import TargetDist
from rt_erg_lib.utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi

import numpy as np


size = 20

ped_state = np.load('sim_data_50.npy')
ped_state = ped_state[500]
print(ped_state.shape)

bounds = [
        np.array([(x, 0) for x in np.linspace(0, size, 1000)]),
        np.array([(x, size) for x in np.linspace(0, size, 1000)]),
        np.array([(0, y) for y in np.linspace(0, size, 1000)]),
        np.array([(size, y) for y in np.linspace(0, size, 1000)])
    ]

env = IntegratorSE2(size=size)
model = IntegratorSE2(size=size)
t_dist = TargetDist(num_pts=50, ped_state=ped_state, bounds=bounds, size=size)

erg_ctrl = RTErgodicControl(model, t_dist, horizon=500, num_basis=100, batch_size=1000)
erg_ctrl.phik = convert_phi2phik(erg_ctrl.basis, t_dist.grid_vals, t_dist.grid)


from tqdm import tqdm
import matplotlib.pyplot as plt
tf = 5000
init_state = np.array([6., 6., 0])

log = {'trajectory' : []}
state = env.reset(init_state)
for t in tqdm(range(tf)):
    ctrl = erg_ctrl(state)
    state = env.step(ctrl)
    log['trajectory'].append(state)
print('doneee')

xy, vals = t_dist.get_grid_spec()
plt.contourf(*xy, vals, levels=20)
xt = np.stack(log['trajectory'])

plt.scatter(xt[:tf,0], xt[:tf,1], s=10)
plt.show()

phi = convert_phik2phi(erg_ctrl.basis, erg_ctrl.phik, t_dist.grid)
plt.contourf(*xy, phi.reshape(50,50), levels=20)
plt.show()


