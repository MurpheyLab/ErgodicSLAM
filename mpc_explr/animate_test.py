import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import time


ft = 100
dt = 2*pi / ft
ts = np.arange(ft) * dt
radius = 4
traj = np.c_[np.sin(ts), np.cos(ts)] * radius

fig = plt.figure()
ax = fig.add_subplot(111)

for t in range(ft):
    ax.cla()
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    robot = traj[t]
    ax.scatter(robot[0], robot[1])

    print(robot)
    plt.pause(0.1)

plt.show()
