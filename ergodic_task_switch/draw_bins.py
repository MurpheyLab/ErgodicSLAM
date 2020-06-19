import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

fig = plt.figure()
ax = fig.add_subplot(111)

def sub_animate(i):
    ax.cla()
    ax.set_title('Iter: {}'.format(i))
    ax.bar(['Task 1\nTime', 'Task 2'], [i**2, 2*i], color=['red', 'blue'])

anim = animation.FuncAnimation(fig, sub_animate, frames=50, interval=50)
plt.show()
