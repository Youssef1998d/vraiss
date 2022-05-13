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
parT=[5.,1.5,1.5,1.3,0.2,0.05]
a = np.linspace(0.5,4.5,2)


def répartition(data, param):
    p0, p1, p2, p3, p4, p5=param
    rep = 1-aut.numpy.exp((p0/p1)*(1-(aut.numpy.exp(-p5*(aut.numpy.exp((p3/data)**p2)-1)**(-p4)))**(-p1)))
    return rep

# print('\n répartition = ',répartition(a, parT))

def densité(x, param):
    a0, a1, a2, a3, a4, a5=param
    den = a0*a2*a4*a5*(a3/x)**a2*aut.numpy.exp((a3/x)**a2)*aut.numpy.exp(a0*(1 - 1/aut.numpy.exp(-a5/(aut.numpy.exp((a3/x)**a2) - 1)**a4)**a1)/a1)/(x*(aut.numpy.exp((a3/x)**a2) - 1)*(aut.numpy.exp((a3/x)**a2) - 1)**a4*aut.numpy.exp(-a5/(aut.numpy.exp((a3/x)**a2) - 1)**a4)**a1)
    return den

print('densité = ',densité(3, parT))
data= densité(3,parT)
print('\n densité = ',densité(3, parT))

# def vraisemblance(param):
#     L = []
#     for i in data:
#         y =  densité(i, param)
#         L.append(y)
#     return np.prod(L)
# print('\n vraisemblance = ',vraisemblance(parT))

# def log_vraissemblance(param):
#     LnL = []
#     for i in data:
#         y =  densité(i, param)
#         LnL.append(np.log(y))
#     return np.sum(LnL)

# print('\n log_vraissemblance = ',log_vraissemblance(parT))

# def grad_f(param):
#     a0, a1, a2, a3, a4, a5=param
#     g = aut.grad
#     return array([g(log_vraissemblance, i)(param) for i in range(6)])
# print(grad_f(parT))

# # print([float(fngrad(v)) for v in parT])
# # # print('\n data = ',data)















