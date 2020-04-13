import numpy as np
import sympy as sym
from sympy import *
import matplotlib.pyplot as plt

def display(expr):
    text = latex(expr)
    plt.text(0, 0.6, '$%s$'.format(text))
    plt.show()

if __name__ == '__main__':
    init_printing(latex=True)
    x, mu, sig = symbols(r'x, mu, sigma')
    # mu = 0
    sig = 2.5
    px = 1 / sqrt(2*pi*sig) * exp(-0.5 * (x-mu)**2 / sig)
    ev = Integral(px * x, (x,-oo,oo))
    pprint(px)
    pprint(ev)
    ev_expr = ev.doit()
    pprint(ev_expr)

    ev_f = lambdify([mu], ev_expr)
    print(ev_f(2))
