from decimal import DivisionByZero
from pkgutil import iter_modules
import sympy as sp
import numpy as np
import mpmath as mp
import os
import time as ti
clear = lambda: os.system('cls')
clear()
from sympy import exp, log, Matrix, Transpose
xx, a0, a1, a2, a3, a4, a5  = sp.var('x, a0, a1, a2, a3, a4, a5', real=True, positive=True);


parG=[a0, a1, a2, a3, a4, a5]
parT=[0.1,0.55,1.3,2.6,2.4,0.02]



def répartition(data, param):
    p0, p1, p2, p3, p4, p5=param
    rep = 1-exp((p0/p1)*(1-(exp(-p5*(exp((p3/data)**p2)-1)**(-p4)))**(-p1)))
    return rep

# print('répartition = ',répartition(xx, parG))

def densité(data, param):
    den=sp.diff(répartition(data, param), data)
    return den#(sp.simplify(den))

print('densité = ',densité(xx, parG))#.subs({xx:3}))

a = np.linspace(0.5,5,15)

def Echantiollon(X, param):
    data=[]
    for i in X:
        y =  densité(xx, param).subs({xx:i})
        data.append(y)
    return (data)
data= Echantiollon(a,parT)
print('\n\n\n data = ',data)

def vraisemblance(param):
    L = []
    for i in data:
        y =  densité(xx, param).subs({xx:i})
        L.append(y)
    return np.prod(L)

def log_vraissemblance(param):
    LnL = []
    for i in data:
        y =  densité(xx, param).subs({xx:i})
        LnL.append(log(y))
    return np.sum(LnL)

LnL = log_vraissemblance(parT)
# print(LnL)

def grad(param):
    p0, p1, p2, p3, p4, p5=param
    GR=[]
    for c in parG:
        Y =  sp.diff(log_vraissemblance(parG),c).subs({a0:p0, a1:p1, a2:p2, a3:p3, a4:p4, a5:p5})
        GR.append(Y)
    return (Matrix(GR))

# print(grad(parT))


def Jacob(param):
    #SYMPY_CACHE_SIZE = 5000
    p0, p1, p2, p3, p4, p5=param
    start = ti.time()
    A=grad(parG)
    end = ti.time();print('1 =',end-start)
    J=A.jacobian(parG).subs({a0:p0, a1:p1, a2:p2, a3:p3, a4:p4, a5:p5})
    start = ti.time();print('2 =',start-end)
    return J
print(Jacob(parT))

# k=0
# iter_max=1
# epsilone=0.01
# norme=epsilone+1
# x=parI=Matrix([0.1,0.55,1.3,2.6,2.4,0.02])

# start = ti.time()
# while norme>epsilone and k<iter_max:
#     # start = ti.time()
#     Jk=Jacob(x)
#     # end = ti.time();print('1 =',end-start)
#     Gk=grad(x)
#     # start = ti.time();print('2 =',start-end)
#     dk=Jk.inv()*Gk
#     # end = ti.time();print('3 =',end-start)
#     Xk=x-dk
#     norme=(Gk).norm()
#     # start = ti.time();print('4 =',start-end)
#     x=Xk
#     print('\n à litération ',k,'la norme = ',norme,'la solution est ',x)
#     k+=1


# end = ti.time()
# print('\n \n Time = ',end - start)


# # print('\n',x)
# # print('\n',Jk)
# # print('\n',Gk)
# # print('\n',dk)
# # print('\n',Xk)
# # print('\n',norme)




