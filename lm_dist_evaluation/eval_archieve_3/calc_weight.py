import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 1000)

def func(x, t):
    return (t / (t + x**3)) ** 3

ts = [1e-01, 1e-02, 1e-03, 5e-04, 1e-04, 1e-05, 1e-06, 1e-07]
for t in ts:
    f = func(x, t)
    plt.plot(x, f)

plt.show()
