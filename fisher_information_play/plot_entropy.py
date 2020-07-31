import numpy as np
from numpy import log, exp
import matplotlib.pyplot as plt


def entropy(c):
    return -c*log(c) - (1-c)*log(1-c)

x = np.linspace(0, 1, 101)
y = entropy(x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y)
ax1.set_xlabel(r'$p(m_i)$')
ax1.set_ylabel(r'Entropy')

plt.show()
