import sys
sys.path.append('..')

from rt_erg_lib.integrator_se2 import IntegratorSE2
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import itertools

size = 20.
robot = np.array([size/2., size/2., 0.])
model = IntegratorSE2(size = size)
actions = Box(np.array([-4., -4.]), np.array([4., 4.]))

nodes = 2
depth = 10

samples = []
for i in range(depth):
    samples.append(np.array([actions.sample() for _ in range(nodes)]))
ctrls = np.array(list(itertools.product(*samples)))
print(ctrls.shape)

trees = []
for branch in ctrls:
    model.reset(robot)
    states = [robot]
    for ctrl in branch:
        states.append(model.step(ctrl))
    trees.append(np.array(states))
trees = np.array(trees)
print(trees.shape)

# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(0, size)
ax1.set_ylim(0, size)
ax1.set_aspect('equal', 'box')

ax1.scatter(robot[0], robot[1], s=20, c='b')

for branch in trees:
    ax1.plot(branch[:,0], branch[:,1], c='r')

plt.show()
