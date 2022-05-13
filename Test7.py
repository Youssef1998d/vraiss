import numpy as np
import autograd as aut
from autograd.numpy import *
# from numpy import autograd
from autograd import grad, jacobian
import mpmath as mp
from decimal import DivisionByZero
from pkgutil import iter_modules
import os
import time as ti
clear = lambda: os.system('cls')
clear()

# parT=[0.1, 0.55, 1.3, 2.6, 2.4, 0.02]
parT=[5.,7.,1.5,1.3,7.2,0.05]
a = np.linspace(0.5,4.5,17)
a0, a1, a2, a3, a4, a5=parT


def répartition(data, p0, p1, p2, p3, p4, p5):
    return 1-aut.numpy.exp((p0/p1)*(1-(aut.numpy.exp(-p5*(aut.numpy.exp((p3/data)**p2)-1)**(-p4)))**(-p1)))

def densité(x, p0, p1, p2, p3, p4, p5):
    # g = aut.grad
    return p0*p2*p4*p5*(p3/x)**p2*aut.numpy.exp((p3/x)**p2)*aut.numpy.exp(p0*(1 - 1/aut.numpy.exp(-p5/(aut.numpy.exp((p3/x)**p2) - 1)**p4)**p1)/p1)/(x*(aut.numpy.exp((p3/x)**p2) - 1)*(aut.numpy.exp((p3/x)**p2) - 1)**p4*aut.numpy.exp(-p5/(aut.numpy.exp((p3/x)**p2) - 1)**p4)**p1)
    # return (array([g(répartition, 0)(x, p0, p1, p2, p3, p4, p5)]))

data=densité(a, a0, a1, a2, a3, a4, a5)
print(densité(a, a0, a1, a2, a3, a4, a5))


def vraisemblance(p0, p1, p2, p3, p4, p5):
    L = []
    for i in data:
        y =  densité(i, p0, p1, p2, p3, p4, p5)
        L.append(y)
    return np.prod(L)
print('\n vraisemblance = ',vraisemblance(a0, a1, a2, a3, a4, a5))

def log_vraissemblance(p0, p1, p2, p3, p4, p5):
    LnL = []
    for i in data:
        y =  densité(i, p0, p1, p2, p3, p4, p5)
        LnL.append(aut.numpy.log(y))
    return np.sum(LnL)

print('\n log_vraissemblance = ',log_vraissemblance(a0, a1, a2, a3, a4, a5))

def grad_LnL(p0, p1, p2, p3, p4, p5):
    g = aut.grad
    if np.isnan(p0): print("p0 nan\n")
    return (array([g(log_vraissemblance, i)(p0, p1, p2, p3, p4, p5)for i in range(6)]).T)

print(grad_LnL(a0, a1, a2, a3, a4, a5))

def J_f(x,y,z,a,b,c):
    j = aut.jacobian
    return array([j(grad_LnL, i)(x,y,z,a,b,c)for i in range(6)])

print(J_f(a0, a1, a2, a3, a4, a5))














