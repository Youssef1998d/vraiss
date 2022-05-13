import numpy as np
import math
import time as ti
from math import *
import random
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal , uniform
import os
import scipy
import scipy.stats
import scipy.optimize as opt
import mpmath as mp
from sympy import symbols, cos, exp, log, sqrt, diff, nsolve
clear = lambda: os.system('cls')
clear()
plt.close()
mu,sigma,data  = symbols('mu,sigma,data ')

# import csv
# with open('Fathi1212.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     X=[]
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             x=row[0]
#             X.append(x)
#             line_count += 1
#     print(f'Processed {line_count} lines.')
# # print('X= ',X,'\n')

def densité(data,a,b):
    return (1/(b*sqrt(2*pi)))*exp((-1/2)*((data-a)/b)**2)
# print(diff(densité(),mu))
# print(diff(densité(),mu).subs({data:1,mu:0,sigma:1}))
# print(densité(data,mu,sigma))
X=np.random.normal(2,3, 100)


def vraisemblance(X,a,b):
    L = []
    for i in X:
        y =  densité(data,a,b).subs({data:i})
        L.append(y)
    return np.prod(L)

# print(vraisemblance(X,mu,sigma))

def log_vraissemblance(X,a,b):
    LnL = []
    for i in X:
        y =  densité(data,a,b).subs({data:i})
        LnL.append(log(y))
    return np.sum(LnL)

# print(log_vraissemblance(X,mu,sigma))



def f(variables) :
    (x,y) = variables
    first_eq = diff(log_vraissemblance(X,mu,sigma),mu).subs({mu:x,sigma:y})
    second_eq = diff(log_vraissemblance(X,mu,sigma),sigma).subs({mu:x,sigma:y})
    return [first_eq, second_eq]

# print(f([1,2]))
first_eq1 = diff(log_vraissemblance(X,mu,sigma),mu)#.subs({mu:x,sigma:y})
second_eq2 = diff(log_vraissemblance(X,mu,sigma),sigma)#.subs({mu:x,sigma:y})
# print('first_eq = ',first_eq1)
# print('second_eq = ',second_eq2)
start = ti.time()
solution1 = opt.fsolve(f, ([0.1, 1])) # fsolve(equations, X_0)

print('solution1 = ',solution1)
end = ti.time()
print(end - start)
start = ti.time()
mp.dps = 15
solution2 = nsolve((first_eq1,second_eq2),(mu,sigma), (0.1, 1), dict=True, prec=3) # fsolve(equations, X_0)

print('solution2 = ',solution2)
end = ti.time()
print('Time = ',end - start)