import numpy as np
import matplotlib.pyplot as plt


xt = np.linspace(0, 2.0, 10000)

def func(x):
    return -1 / np.log(x/2)

yt = func(xt)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xt, yt, s=1)

print(func(1.5))
plt.show()
