from qutip import *
from pylab import *
from scipy import *
import time
# Define paramters
N = 500  # number of basis states to consider
a = destroy(N)
H = a.dag() * a
psi0 = basis(N, 10)  # initial state
kappa = 0.1  # coupling to oscillator

# collapse operators
c_op_list = []
n_th_a = 2  # temperature with average of 2 excitations
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_op_list.append(sqrt(rate) * a)  # decay operators
rate = kappa * n_th_a
if rate > 0.0:
    c_op_list.append(sqrt(rate) * a.dag())  # excitation operators

# find steady-state solution
out1 = steadystate(H, c_op_list, method='iterative', use_rcm=0, verbose=1)
print()
s=time.time()
out2 = steadystate(H, c_op_list, method='iterative', use_rcm=1, verbose=1)
