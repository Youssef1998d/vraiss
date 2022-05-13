import math
import numpy as np
import autograd as aut
from autograd.numpy import *

def f(x, y, z, a,b,c):
    return aut.numpy.exp(b*c*(x**2 + 2.0 * y**2+ x*y*aut.numpy.exp(z))**a)


# def f(data, p0, p1, p2, p3, p4, p5):
#     return 1-aut.numpy.exp((p0/p1)*(1-(aut.numpy.exp(-p5*(aut.numpy.exp((p3/data)**p2)-1)**(-p4)))**(-p1)))

# def f(x, y):
#     return array([1.0 * x + 2.0 * aut.numpy.exp(y), 3.0 * x + 4.0 * y])

print(f(1.,2.,1.,2.,1.,2.))

def grad_f(x,y,z,a,b,c):
    g = aut.grad
    return (array([g(f, i)(x,y,z,a,b,c)for i in range(6)]))
print(grad_f(1.0, 2.0,1.,2.,1.,2.))



def J_f(x,y,z,a,b,c):
    j = aut.jacobian
    return array([j(grad_f, i)(x,y,z,a,b,c)for i in range(6)])
print(J_f(1.,2.,1.,2.,1.,2.))
