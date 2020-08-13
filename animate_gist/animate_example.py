"""
A simple tutorial about generate animations using matplotlib
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import animation

# First generate & store the trajectories (here two circles)
traj_1 = np.stack([ np.sin(np.linspace(0, 2*pi, 100)), np.cos(np.linspace(0, 2*pi, 100)) ])
traj_1 = traj_1.T # transpose so each row corresponds to a time step
traj_2 = np.stack([ np.cos(np.linspace(0, 2*pi, 100)), np.sin(np.linspace(0, 2*pi, 100)) ])
traj_2 = traj_2.T

# Create figure (basis of any matplotlib plot)
#  ... and create axes (basis of any plot operation)
fig = plt.figure()
ax = fig.add_subplot(111) # this case only one plot
ax.set_aspect('equal')
ax.set_title('animation test')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# To accelerate plot, create plot(trajectory) objects before
#  ... the animation begins
traj_1_plot = ax.scatter([], [], s=10, c='r') # don't need to fill in any data now
traj_2_plot = ax.scatter([], [], s=10, c='b')

# Create animateion function, defines what to do in each time step
def animate_at_t(t, traj_1, traj_2, traj_1_plot, traj_2_plot, show_traj=True):
    # fill trajectory at time t into existing plot objects
    #  ... no need to configure color and size now

    if show_traj == False:
        # case 1: plot only current position
        traj_1_plot.set_offsets(traj_1[t])
        traj_2_plot.set_offsets(traj_2[t])
    else:
        # case 2: plot all previoys trajectory
        traj_1_plot.set_offsets(traj_1[:t])
        traj_2_plot.set_offsets(traj_2[:t])

    # must returnn a list of plot objects
    return [traj_1_plot, traj_2_plot]

# Finally, start animation
animate = animation.FuncAnimation(fig, animate_at_t, frames=100, interval=100, fargs=(traj_1, traj_2, traj_1_plot, traj_2_plot, True), repeat=True)

# If needed, save the animation into a GIF (or other formats)
animate.save('animation.gif', writer='imagemagick', fps=10) # need to install imagemagick

plt.show()
