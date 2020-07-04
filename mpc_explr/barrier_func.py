import numpy as np
import matplotlib.pyplot as plt


xt = np.linspace(0, 2.5, 10000)

def func(x, n):
    return 0.00001 * (n**(x) - 1)

yt = func(xt, 50)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xt, yt, s=1)

print(func(1.0, 50))
print(func(2.5, 50))
plt.grid()
plt.show()
