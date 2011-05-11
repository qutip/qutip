#
# Textbook example: Rabi oscillation in the dissipative Jaynes-Cummings model.
# 
#
from qutip import *
import time
from scipy import *
from matplotlib import pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.ticker as ticker
def jc_integrate(N, wc, wa, g, kappa, gamma, psi0, use_rwa, tlist):

    # Hamiltonian
    idc = qeye(N)
    ida = qeye(2)

    a  = tensor(destroy(N), ida)
    sm = tensor(idc, destroy(2))

    if use_rwa: 
        # use the rotating wave approxiation
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
    else:
        H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())
        
    # collapse operators
    c_op_list = []

    n_th_a = 0.0 # zero temperature

    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)

    # evolve and calculate expectation values
    #expt_list = me_ode_solve(tlist, H, psi0, c_op_list, [a.dag() * a, sm.dag() * sm])  

     #or use the MC solver
    ntraj = 10
    Heff = H
    for c_op in c_op_list:
        Heff += - 0.5 * 1j * c_op.dag() * c_op  
    psi = mcsolve(Heff, psi0, tlist, ntraj, c_op_list, [])
    psi_avg = sum(psi,axis=0)/ntraj

    return psi_avg
    
#
# set up the calculation
#
wc = 1.0 * 2 * pi   # cavity frequency
wa = 1.0 * 2 * pi   # atom frequency
g  = 0.05 * 2 * pi  # coupling strength

kappa = 0.005       # cavity dissipation rate
gamma = 0.05        # atom dissipation rate

N = 5               # number of cavity fock states

use_rwa = True

# intial state
psi0 = tensor(basis(N,0),    basis(2,1))    # start with an excited atom 
#psi0 = tensor(coherent(N,1), basis(2,0))   # or a coherent state the in cavity

tlist = linspace(0,25,100)

start_time = time.time()
psi = jc_integrate(N, wc, wa, g, kappa, gamma, psi0, use_rwa, tlist)
print 'time elapsed = ' +str(time.time() - start_time) 
qubit_rhos=ptrace(psi[:],1)

rx=expect(sigmax(),qubit_rhos)
ry=expect(sigmay(),qubit_rhos)
rz=expect(sigmaz(),qubit_rhos)
print rx
print ry
print rz











